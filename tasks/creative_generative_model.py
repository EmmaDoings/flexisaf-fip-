"""A small local generative model for creative-industry idea generation.

The model uses a word-level Markov chain trained on short seed examples from
advertising, film, fashion, and music. It demonstrates how generative AI can
support creative industries by producing draft taglines, story concepts,
moodboards, and lyric fragments without calling an external API.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from typing import DefaultDict


TRAINING_TEXT = {
    "advertising": [
        "A bold campaign where city lights turn every commute into a launch moment for dreamers.",
        "Fresh energy in every bottle, crafted for makers who chase tomorrow before sunrise.",
        "Wear the future today with clean lines, fearless color, and effortless confidence.",
        "The smart home that listens softly, saves energy, and makes every room feel alive.",
        "For creators on the move, power that fits your pocket and keeps ideas flowing.",
    ],
    "film": [
        "A quiet archivist discovers that forgotten songs can unlock memories hidden across the city.",
        "Two rival designers must rebuild a theatre before opening night while a storm rewrites their plans.",
        "In a floating market above the clouds, a young chef trades recipes for clues about home.",
        "A documentary crew follows street dancers turning abandoned stations into stages of resistance.",
        "When the moon disappears, a radio host gathers strangers through stories broadcast after midnight.",
    ],
    "fashion": [
        "A coastal collection of sand linen, glass blue silk, pearl buttons, and wind-shaped silhouettes.",
        "Streetwear inspired by night markets, reflective trims, oversized pockets, and neon embroidery.",
        "Minimal tailoring with warm clay tones, recycled cotton, sculptural sleeves, and quiet luxury.",
        "Festival looks mixing sunlit crochet, botanical prints, metallic sandals, and playful layered textures.",
        "A futuristic capsule wardrobe of chrome jackets, matte black knits, and modular travel accessories.",
    ],
    "music": [
        "Under violet speakers we dance through static dreams and let the sunrise remix our names.",
        "Your shadow keeps the rhythm while my heartbeat samples rain on the window.",
        "Golden drums in the alley call us back to nights we never finished singing.",
        "I keep a chorus in my jacket for the days when silence gets too heavy.",
        "Neon tides pull us closer as the bassline paints the harbor blue.",
    ],
}

START = "<START>"
END = "<END>"


def tokenize(text: str) -> list[str]:
    """Split text into words while keeping simple punctuation as tokens."""
    return re.findall(r"[A-Za-z']+|[.,]", text)


def untokenize(tokens: list[str]) -> str:
    sentence = " ".join(tokens)
    sentence = re.sub(r"\s+([.,])", r"\1", sentence)
    return sentence[:1].upper() + sentence[1:]


class MarkovCreativeModel:
    def __init__(self, examples: list[str], order: int = 2) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")

        self.order = order
        self.transitions: DefaultDict[tuple[str, ...], list[str]] = defaultdict(list)
        self._train(examples)

    def _train(self, examples: list[str]) -> None:
        for example in examples:
            tokens = [START] * self.order + tokenize(example) + [END]
            for index in range(len(tokens) - self.order):
                state = tuple(tokens[index : index + self.order])
                next_token = tokens[index + self.order]
                self.transitions[state].append(next_token)

    def generate(self, max_words: int = 28) -> str:
        state = (START,) * self.order
        output: list[str] = []

        for _ in range(max_words):
            choices = self.transitions.get(state)
            if not choices:
                break

            next_token = random.choice(choices)
            if next_token == END:
                break

            output.append(next_token)
            state = (*state[1:], next_token)

        return untokenize(output)


def build_model(use_case: str, order: int) -> MarkovCreativeModel:
    examples = TRAINING_TEXT[use_case]
    return MarkovCreativeModel(examples, order=order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate creative-industry concepts with a local Markov model."
    )
    parser.add_argument(
        "--use-case",
        choices=sorted(TRAINING_TEXT),
        default="advertising",
        help="Creative-industry area to practise.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of ideas to generate.")
    parser.add_argument("--order", type=int, default=2, help="Number of previous words used as context.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model = build_model(args.use_case, args.order)
    title = args.use_case.replace("_", " ").title()
    print(f"Creative Generative AI Model: {title}")
    print("-" * 45)

    for index in range(1, args.count + 1):
        print(f"{index}. {model.generate()}")


if __name__ == "__main__":
    main()
