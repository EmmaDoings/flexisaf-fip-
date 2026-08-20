# TODO - Admissions Dashboard

## Step 1: Repo understanding
- [x] Inspect existing files
- [x] Read `Project/application.csv` header to confirm available columns

## Step 2: Implement dashboard
- [x] Create `Project/admissions_dashboard.py`
  - [x] CLI args: `--csv` and `--outdir`
  - [x] Robust CSV loading + error handling
  - [x] Cleaning: missing data handling + duplicates
  - [x] Compute acceptance rate + outcome distribution
  - [x] Compute GPA/SAT/Extracurricular histograms
  - [x] Export `summary.csv` and `summary.html`
  - [x] Rule-based narrative insight generator
  - [x] Skip gender/program sections when columns are not present (dataset has only GPA/SAT/Extracurricular/Admission_Status)


## Step 3: GitHub-ready project files
- [x] Create `README.md` with run instructions + description of outputs
- [x] Create `requirements.txt`
- [x] Create `.gitignore`


## Step 4: Runtime validation
- [x] Run the script and ensure outputs are created under `Project/outputs`


## Step 5: GitHub push (user-side)
- [ ] Initialize git, commit, and push (commands provided after validation)


