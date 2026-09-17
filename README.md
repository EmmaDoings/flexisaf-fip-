# Admissions EDA Dashboard (Python)

This project builds an Exploratory Data Analysis (EDA) dashboard for applicant/admission datasets using **Pandas**, **Matplotlib**, and **Seaborn**.

## What this repo includes
- `Project/admissions_dashboard.py`: loads and cleans a CSV, computes admission metrics, generates charts, and exports summary reports.
- `Project/application.csv`: the provided dataset.

## Data expectations
The current `Project/application.csv` has columns:
- `GPA`
- `SAT_Score`
- `Extracurricular_Activities`
- `Admission_Status` (e.g., `Accepted`, `Rejected`, `Waitlisted`)

If the dataset is missing additional columns (like gender or program preference), the script will **skip** those analyses and clearly note what’s unavailable.

## Run
From the repo root:

```bash
python Project/admissions_dashboard.py --csv Project/application.csv --outdir Project/outputs
```

Outputs will be generated in `Project/outputs/`, including:
- `summary.csv`
- `summary.html`
- chart images (PNG)

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes on the “AI-powered” insight
This implementation uses a **rule-based narrative template** (no external API).

## Intelligent Class Summarizer

`Project/class_summarizer.py` processes `.txt` or `.csv` class transcripts, extracts topic clusters, creates topic-wise summaries, detects next steps/assignments, and exports structured reports.

Run the included sample:

```bash
python Project/class_summarizer.py --input Project/sample_transcript.txt --outdir Project/outputs/class_summary --topics 3
```

Outputs:
- `Project/outputs/class_summary/class_summary.html`
- `Project/outputs/class_summary/class_summary.csv`

For CSV files, include a text-like column such as `message`, `text`, `transcript`, `chat`, `content`, or `utterance`.

The implementation uses local topic modeling with scikit-learn LDA and TF-IDF extractive summarization, so it works offline without an API key.

## Image deblurring app

The app in `tasks/Image deblurring app/model.py` trains a compact residual
MobileNetV3 model on paired GoPro images and provides a Streamlit interface for
testing images from a physical camera. Arrange the downloaded dataset as:

```text
gopro/
	blur/
	sharp/
```

Install the additional dependencies, train, and launch the app from the repo root:

```bash
pip install -r requirements.txt
python "tasks/Image deblurring app/model.py" train --root gopro --epochs 10 --output deblurrer.pt
streamlit run "tasks/Image deblurring app/model.py"
```

The model uses paired L1 training and the reusable `psnr()` function supports
evaluation on a held-out GoPro split. The MobileNetV3 backbone keeps inference
lightweight enough for physical-camera testing.

## Creative generative AI model

`tasks/creative_generative_model.py` is a local word-level Markov model that
practises generative AI uses in creative industries. It can generate draft
advertising concepts, film premises, fashion moodboards, and music lyric ideas.

Run examples from the repo root:

```bash
python tasks/creative_generative_model.py --use-case advertising --count 5
python tasks/creative_generative_model.py --use-case film --count 3 --seed 12
python tasks/creative_generative_model.py --use-case fashion --order 1
python tasks/creative_generative_model.py --use-case music --count 4
```

The script uses only the Python standard library and does not require an API key.

