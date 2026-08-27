"""GoPro image deblurring app and training utilities.

The app accepts paired GoPro-style folders (blur/ and sharp/), evaluates PSNR,
and can run inference on images captured by a physical camera or phone.
"""

from __future__ import annotations

import argparse
import io
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MAX_SIDE = 1024


def psnr(prediction: Tensor, target: Tensor) -> float:
	"""Calculate peak signal-to-noise ratio for tensors in [0, 1]."""
	mse = torch.mean((prediction.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
	return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


class GoProPairs(Dataset):
	"""Load paired images from GoPro's blur/sharp directory convention."""

	def __init__(self, root: str | Path, crop_size: int = 256, training: bool = True):
		self.root = Path(root)
		self.crop_size = crop_size
		self.training = training
		blur_root = self.root / "blur"
		sharp_root = self.root / "sharp"
		if not blur_root.exists() or not sharp_root.exists():
			raise FileNotFoundError("Dataset must contain sibling 'blur' and 'sharp' folders.")
		sharp_files = {path.relative_to(sharp_root): path for path in sharp_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS}
		self.pairs = [
			(path, sharp_files[relative])
			for path in blur_root.rglob("*")
			if path.suffix.lower() in IMAGE_EXTENSIONS
			for relative in [path.relative_to(blur_root)]
			if relative in sharp_files
		]
		if not self.pairs:
			raise ValueError("No matching blur/sharp image pairs were found.")

	def __len__(self) -> int:
		return len(self.pairs)

	def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
		blur_path, sharp_path = self.pairs[index]
		blur = Image.open(blur_path).convert("RGB")
		sharp = Image.open(sharp_path).convert("RGB")
		if self.training:
			blur, sharp = self._random_crop_pair(blur, sharp)
			if random.random() > 0.5:
				blur, sharp = blur.transpose(Image.Transpose.FLIP_LEFT_RIGHT), sharp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
		else:
			blur, sharp = self._fit(blur), self._fit(sharp)
		return TF.to_tensor(blur), TF.to_tensor(sharp)

	def _random_crop_pair(self, blur: Image.Image, sharp: Image.Image) -> tuple[Image.Image, Image.Image]:
		width, height = blur.size
		if width < self.crop_size or height < self.crop_size:
			scale = self.crop_size / min(width, height)
			size = (max(self.crop_size, int(width * scale)), max(self.crop_size, int(height * scale)))
			blur = blur.resize(size, Image.Resampling.BICUBIC)
			sharp = sharp.resize(size, Image.Resampling.BICUBIC)
			width, height = blur.size
		left = random.randint(0, width - self.crop_size)
		top = random.randint(0, height - self.crop_size)
		box = (left, top, left + self.crop_size, top + self.crop_size)
		return blur.crop(box), sharp.crop(box)

	def _fit(self, image: Image.Image) -> Image.Image:
		width, height = image.size
		scale = min(1, self.crop_size / max(width, height))
		return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)


class MobileNetDeblurrer(nn.Module):
	"""Lightweight residual deblurrer using MobileNetV3 for physical testing."""

	def __init__(self, pretrained: bool = False):
		super().__init__()
		weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
		backbone = mobilenet_v3_small(weights=weights).features
		self.encoder = nn.Sequential(*list(backbone.children())[:5])
		with torch.inference_mode():
			feature_channels = self.encoder(torch.zeros(1, 3, 64, 64)).shape[1]
		self.head = nn.Sequential(
			nn.Conv2d(feature_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(32, 3, 3, padding=1),
		)

	def forward(self, image: Tensor) -> Tensor:
		features = self.encoder(image)
		restored = self.head(features)
		restored = torch.nn.functional.interpolate(restored, size=image.shape[-2:], mode="bilinear", align_corners=False)
		return (image + restored).clamp(0, 1)


def train(root: str, output: str = "deblurrer.pt", epochs: int = 5, batch_size: int = 4, learning_rate: float = 2e-4) -> None:
	"""Train on paired GoPro images and save a portable checkpoint."""
	dataset = GoProPairs(root, training=True)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MobileNetDeblurrer(pretrained=True).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	loss_fn = nn.L1Loss()
	model.train()
	for epoch in range(epochs):
		losses = []
		for blur, sharp in loader:
			blur, sharp = blur.to(device), sharp.to(device)
			optimizer.zero_grad(set_to_none=True)
			loss = loss_fn(model(blur), sharp)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		print(f"epoch {epoch + 1}/{epochs} - L1: {np.mean(losses):.4f}")
	torch.save({"model": model.state_dict(), "architecture": "mobilenet_v3_small_residual"}, output)
	print(f"Saved checkpoint to {output}")


def evaluate(root: str, checkpoint: str, crop_size: int = 512) -> None:
	"""Evaluate a checkpoint on paired blur/sharp images with PSNR."""
	dataset = GoProPairs(root, crop_size=crop_size, training=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = _load_checkpoint(checkpoint).to(device)
	model.eval()
	scores = []
	for blur, sharp in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
		blur, sharp = blur.to(device), sharp.to(device)
		with torch.inference_mode():
			restored = model(blur)
		scores.append(psnr(restored.cpu(), sharp.cpu()))
	print(f"Evaluated {len(scores)} image pairs")
	print(f"Mean PSNR: {np.mean(scores):.2f} dB")


def predict(input_path: str, output_path: str, checkpoint: str, max_side: int = DEFAULT_MAX_SIDE) -> None:
	"""Deblur one physical test image from a GoPro, phone, or webcam."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = _load_checkpoint(checkpoint, device=device, optimize=True).eval()
	with Path(input_path).open("rb") as image_file:
		source, restored, elapsed_ms = restore_image(model, image_file.read(), device=device, max_side=max_side)
	restored.save(output_path)
	print(f"Saved restored image to {output_path} ({source.width}x{source.height}, {elapsed_ms:.0f} ms)")


@st.cache_resource
def load_model(checkpoint: str, device: str, optimize: bool) -> MobileNetDeblurrer:
	return _load_checkpoint(checkpoint, device=device, optimize=optimize)


def _load_checkpoint(checkpoint: str, device: str = "cpu", optimize: bool = False) -> MobileNetDeblurrer:
	if device == "cpu":
		torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
	model = MobileNetDeblurrer().eval()
	state = torch.load(checkpoint, map_location="cpu", weights_only=True)
	model.load_state_dict(state["model"] if "model" in state else state)
	model = model.to(device).to(memory_format=torch.channels_last)
	if optimize and hasattr(torch, "compile"):
		try:
			model = torch.compile(model, mode="reduce-overhead")
		except Exception:
			pass
	return model


def restore_image(model: nn.Module, uploaded: bytes, device: str = "cpu", max_side: int = DEFAULT_MAX_SIDE) -> tuple[Image.Image, Image.Image, float]:
	source = Image.open(io.BytesIO(uploaded)).convert("RGB")
	image = _resize_for_inference(source, max_side)
	tensor = TF.to_tensor(image).unsqueeze(0).to(device).to(memory_format=torch.channels_last)
	started = time.perf_counter()
	with torch.inference_mode():
		with torch.autocast(device_type="cuda", enabled=device == "cuda"):
			restored = model(tensor)[0].detach().float().cpu()
	elapsed_ms = (time.perf_counter() - started) * 1000
	restored_image = TF.to_pil_image(restored)
	if restored_image.size != source.size:
		restored_image = restored_image.resize(source.size, Image.Resampling.BICUBIC)
	return source, restored_image, elapsed_ms


def _resize_for_inference(image: Image.Image, max_side: int) -> Image.Image:
	if max_side <= 0:
		return image
	width, height = image.size
	scale = min(1, max_side / max(width, height))
	if scale == 1:
		return image
	return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)


def app() -> None:
	st.set_page_config(page_title="ClearFrame", page_icon="◈", layout="wide")
	st.title("ClearFrame")
	st.caption("Fast GoPro-trained image deblurring with a compact MobileNetV3 inference path")
	checkpoint = st.sidebar.text_input("Checkpoint", "deblurrer.pt")
	device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
	device = st.sidebar.selectbox("Inference device", device_options, index=len(device_options) - 1)
	max_side = st.sidebar.slider("Speed / quality limit", 512, 2048, DEFAULT_MAX_SIDE, 128, help="Images are processed at this maximum side length, then resized back for download.")
	optimize = st.sidebar.checkbox("Compile model for speed", value=False, help="Can improve repeated inference after a slower first run on supported PyTorch versions.")
	st.sidebar.markdown("Use `1024` for responsive demos. Increase for quality, decrease for speed.")
	uploaded = st.file_uploader("Upload a blurry camera image", type=sorted(extension.lstrip(".") for extension in IMAGE_EXTENSIONS))
	if uploaded is None:
		st.info("Upload a JPG, PNG, or WebP image to deblur it in the browser.")
		return
	if not Path(checkpoint).exists():
		st.warning(f"Checkpoint '{checkpoint}' was not found. Train one with: python model.py train --root <gopro-folder>")
		return
	model = load_model(checkpoint, device, optimize)
	with st.spinner("Restoring image..."):
		source, restored, elapsed_ms = restore_image(model, uploaded.getvalue(), device=device, max_side=max_side)
	metric_a, metric_b, metric_c = st.columns(3)
	metric_a.metric("Inference", f"{elapsed_ms:.0f} ms")
	metric_b.metric("Device", device.upper())
	metric_c.metric("Input", f"{source.width} x {source.height}")
	original, result = st.columns(2)
	original.image(source, caption="Camera input", use_container_width=True)
	result.image(restored, caption="Deblurred output", use_container_width=True)
	st.download_button("Download restored image", data=_image_bytes(restored), file_name="deblurred.png", mime="image/png")


def _image_bytes(image: Image.Image) -> bytes:
	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return buffer.getvalue()


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the GoPro MobileNet deblurrer or launch its app.")
	subparsers = parser.add_subparsers(dest="command")
	train_parser = subparsers.add_parser("train")
	train_parser.add_argument("--root", required=True, help="Folder containing blur/ and sharp/ subfolders")
	train_parser.add_argument("--output", default="deblurrer.pt")
	train_parser.add_argument("--epochs", type=int, default=5)
	train_parser.add_argument("--batch-size", type=int, default=4)
	train_parser.add_argument("--learning-rate", type=float, default=2e-4)
	evaluate_parser = subparsers.add_parser("evaluate")
	evaluate_parser.add_argument("--root", required=True, help="Folder containing blur/ and sharp/ subfolders")
	evaluate_parser.add_argument("--checkpoint", default="deblurrer.pt")
	evaluate_parser.add_argument("--crop-size", type=int, default=512)
	predict_parser = subparsers.add_parser("predict")
	predict_parser.add_argument("--input", required=True, help="Blurry physical test image")
	predict_parser.add_argument("--output", default="deblurred.png")
	predict_parser.add_argument("--checkpoint", default="deblurrer.pt")
	predict_parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
	args = parser.parse_args()
	if args.command == "train":
		train(args.root, args.output, args.epochs, args.batch_size, args.learning_rate)
	elif args.command == "evaluate":
		evaluate(args.root, args.checkpoint, args.crop_size)
	elif args.command == "predict":
		predict(args.input, args.output, args.checkpoint, args.max_side)
	else:
		app()


if __name__ == "__main__":
	main()
