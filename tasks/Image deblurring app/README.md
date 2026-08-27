# Image Deblurring App

ClearFrame is a lightweight image deblurring project that trains on paired GoPro-style blur/sharp images and uses a compact MobileNetV3 residual model for fast physical testing on camera, phone, or webcam photos.

## Dataset

Use the Kaggle collection: [A Curated List of Image Deblurring Datasets](https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets).

Prepare a folder with matching relative paths under `blur/` and `sharp/`:

```text
data/gopro/
  blur/
    image_001.png
  sharp/
    image_001.png
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from https://pytorch.org/get-started/locally/.

## Train

```bash
python model.py train --root data/gopro --output deblurrer.pt --epochs 5 --batch-size 4
```

The training loop uses random crops, flips, L1 reconstruction loss, and saves a portable checkpoint.

## Evaluate PSNR

```bash
python model.py evaluate --root data/gopro --checkpoint deblurrer.pt
```

PSNR is reported in decibels across paired blur/sharp images.

## Physical Testing

Run inference on one image captured from a GoPro, phone, or webcam:

```bash
python model.py predict --input samples/blurry.jpg --output deblurred.png --checkpoint deblurrer.pt
```

Use `--max-side` to trade quality for speed during physical tests:

```bash
python model.py predict --input samples/blurry.jpg --output deblurred.png --checkpoint deblurrer.pt --max-side 768
```

## Web App

```bash
streamlit run model.py
```

Upload a blurry image, compare the original and restored output, then download the result.

The sidebar includes inference controls:

- `Inference device`: uses CUDA automatically when available.
- `Speed / quality limit`: resizes large camera images before inference, then restores the output to the original dimensions.
- `Compile model for speed`: optionally enables `torch.compile` for faster repeated inference after a slower first run.

## Deployment

For Streamlit Community Cloud:

1. Push this folder to GitHub.
2. Set the app entry point to `tasks/Image deblurring app/model.py` if deploying from the repository root.
3. Keep `deblurrer.pt` in the app folder or provide it through your deployment storage.
4. Use `requirements.txt` as the dependency file.

For platforms that support Procfiles, the included `Procfile` starts the app with:

```bash
streamlit run model.py --server.port=$PORT --server.address=0.0.0.0
```

## Implementation Notes

- `GoProPairs` loads paired images from `blur/` and `sharp/` folders.
- `MobileNetDeblurrer` uses MobileNetV3-Small features and predicts a residual correction.
- `psnr` provides a standard quality metric for paired evaluation.
- Inference uses cached model loading, `torch.inference_mode`, channels-last tensors, optional `torch.compile`, CUDA autocast, and image downscaling for responsive demos.
