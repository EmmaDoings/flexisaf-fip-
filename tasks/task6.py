import re
from typing import List


SAMPLE_TEXTS: List[str] = [
    "I love NLP! It is amazing  and super fun!!!",
    "The weather is bad... and the train was late @station https://example.com",
    "Can we meet tomorrow? Sure, let's do it!!",
    "This product is great, but the service was terrible :(",
]


def clean_text(text: str) -> str:
    """Apply three common text-cleaning steps: lower casing, punctuation removal, and frequent-word removal."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_frequent_words(text: str, stop_words: set[str]) -> str:
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)


def main() -> None:
    frequent_words = {
        "the",
        "is",
        "and",
        "it",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "for",
        "was",
        "were",
        "this",
        "that",
        "but",
        "with",
        "we",
        "can",
        "let",
        "s",
        "do",
        "i",
        "my",
        "our",
    }

    print("Sample texts:")
    for index, text in enumerate(SAMPLE_TEXTS, start=1):
        print(f"{index}. {text}")

    print("\nCleaned texts (lowercasing + punctuation removal + frequent word removal):")
    for text in SAMPLE_TEXTS:
        cleaned = clean_text(text)
        cleaned = remove_frequent_words(cleaned, frequent_words)
        print(cleaned)


if __name__ == "__main__":
    main()
