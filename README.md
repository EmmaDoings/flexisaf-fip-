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

