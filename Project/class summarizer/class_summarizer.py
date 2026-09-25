import argparse
import csv
import html
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


STOPWORDS = "english"
ACTION_PATTERNS = re.compile(
    r"\b(homework|assignment|assign|submit|deadline|due|next step|review|read|practice|complete|prepare|task)\b",
    re.IGNORECASE,
)


@dataclass
class TopicSummary:
    topic_id: int
    label: str
    keywords: List[str]
    messages: List[str]
    summary: str
    next_steps: List[str]
    assignments: List[str]


def load_transcript(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return [_clean_message(line) for line in lines if _clean_message(line)]

    if ext == ".csv":
        df = pd.read_csv(path)
        if df.empty:
            return []

        text_col = _pick_text_column(df)
        if not text_col:
            raise ValueError(
                "CSV must include a text-like column such as message, text, transcript, chat, or content."
            )

        return [
            _clean_message(str(value))
            for value in df[text_col].dropna().tolist()
            if _clean_message(str(value))
        ]

    raise ValueError("Unsupported transcript format. Use .txt or .csv")


def _pick_text_column(df: pd.DataFrame) -> str:
    preferred = ["message", "text", "transcript", "chat", "content", "utterance"]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    if not object_cols:
        return ""
    return max(object_cols, key=lambda col: df[col].astype(str).str.len().mean())


def _clean_message(text: str) -> str:
    text = re.sub(r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_topic_summaries(messages: Sequence[str], topic_count: int) -> List[TopicSummary]:
    if not messages:
        raise ValueError("Transcript has no usable messages.")

    topic_count = max(1, min(topic_count, len(messages)))
    vectorizer = CountVectorizer(stop_words=STOPWORDS, min_df=1, ngram_range=(1, 2))
    doc_term = vectorizer.fit_transform(messages)

    if doc_term.shape[1] == 0:
        raise ValueError("Transcript does not contain enough meaningful words to summarize.")

    lda = LatentDirichletAllocation(n_components=topic_count, random_state=42)
    topic_scores = lda.fit_transform(doc_term)
    assignments = topic_scores.argmax(axis=1)
    feature_names = vectorizer.get_feature_names_out()

    summaries: List[TopicSummary] = []
    for topic_id in range(topic_count):
        topic_messages = [msg for index, msg in enumerate(messages) if assignments[index] == topic_id]
        if not topic_messages:
            continue

        keywords = _topic_keywords(lda.components_[topic_id], feature_names, limit=6)
        label = ", ".join(keywords[:3]).title() if keywords else f"Topic {topic_id + 1}"
        summary = summarize_messages(topic_messages)
        steps, homework = extract_actions(topic_messages)

        summaries.append(
            TopicSummary(
                topic_id=topic_id + 1,
                label=label,
                keywords=keywords,
                messages=topic_messages,
                summary=summary,
                next_steps=steps,
                assignments=homework,
            )
        )

    return sorted(summaries, key=lambda item: len(item.messages), reverse=True)


def _topic_keywords(topic_weights, feature_names, limit: int) -> List[str]:
    top_indexes = topic_weights.argsort()[-limit:][::-1]
    return [feature_names[index] for index in top_indexes]


def summarize_messages(messages: Sequence[str], max_sentences: int = 3) -> str:
    sentences = _split_sentences(" ".join(messages))
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words=STOPWORDS)
    tfidf = vectorizer.fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    ranked = sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)
    chosen = sorted(ranked[:max_sentences])
    return " ".join(sentences[index] for index in chosen)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def extract_actions(messages: Sequence[str]) -> Tuple[List[str], List[str]]:
    next_steps: List[str] = []
    assignments: List[str] = []

    for sentence in _split_sentences(" ".join(messages)):
        if not ACTION_PATTERNS.search(sentence):
            continue
        cleaned = sentence.strip(" -")
        target = assignments if re.search(r"\b(homework|assignment|submit|due|deadline)\b", cleaned, re.IGNORECASE) else next_steps
        if cleaned not in target:
            target.append(cleaned)

    if not next_steps:
        next_steps.append("Review the topic summary and clarify any unresolved questions in the next class.")
    if not assignments:
        assignments.append("No explicit assignment detected for this topic.")

    return next_steps[:5], assignments[:5]


def export_reports(summaries: Sequence[TopicSummary], outdir: str, source_path: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "class_summary.csv")
    html_path = os.path.join(outdir, "class_summary.html")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "keywords", "message_count", "summary", "next_steps", "assignments"])
        for item in summaries:
            writer.writerow(
                [
                    item.label,
                    ", ".join(item.keywords),
                    len(item.messages),
                    item.summary,
                    " | ".join(item.next_steps),
                    " | ".join(item.assignments),
                ]
            )

    cards = "\n".join(_render_topic_card(item) for item in summaries)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Intelligent Class Summary</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #eef3fb; }}
    header {{ padding: 32px; background: #172033; color: white; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    .card {{ background: white; border-radius: 14px; box-shadow: 0 8px 28px rgba(23,32,51,.08); margin: 0 0 18px; padding: 22px; }}
    .meta {{ color: #637083; font-size: 14px; }}
    .pill {{ display: inline-block; margin: 3px 4px 3px 0; padding: 5px 9px; background: #dfeaff; border-radius: 999px; font-size: 13px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>Intelligent Class Summary</h1>
    <p>Source: {html.escape(source_path)}</p>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def _render_topic_card(item: TopicSummary) -> str:
    keywords = "".join(f"<span class='pill'>{html.escape(word)}</span>" for word in item.keywords)
    next_steps = "".join(f"<li>{html.escape(step)}</li>" for step in item.next_steps)
    assignments = "".join(f"<li>{html.escape(task)}</li>" for task in item.assignments)
    return f"""
<section class="card">
  <p class="meta">Topic {item.topic_id} - {len(item.messages)} message(s)</p>
  <h2>{html.escape(item.label)}</h2>
  <div>{keywords}</div>
  <h3>Summary</h3>
  <p>{html.escape(item.summary)}</p>
  <h3>Next Steps</h3>
  <ul>{next_steps}</ul>
  <h3>Assignments</h3>
  <ul>{assignments}</ul>
</section>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic-wise class transcript summarizer")
    parser.add_argument("--input", required=True, help="Path to .txt or .csv transcript")
    parser.add_argument("--outdir", default="outputs", help="Directory for HTML/CSV reports")
    parser.add_argument("--topics", type=int, default=4, help="Number of topics to extract")
    args = parser.parse_args()

    messages = load_transcript(args.input)
    summaries = build_topic_summaries(messages, args.topics)
    export_reports(summaries, args.outdir, args.input)

    print("Class summary generated.")
    print(f"- Messages processed: {len(messages)}")
    print(f"- Topics created: {len(summaries)}")
    print(f"- Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()
