# TODO - Admissions Dashboard

## Step 1: Repo understanding
- [x] Inspect existing files
- [x] Read `Project/application.csv` header to confirm available columns

## Step 2: Implement dashboard
- [ ] Create `Project/admissions_dashboard.py`
  - [ ] CLI args: `--csv` and `--outdir`
  - [ ] Robust CSV loading + error handling
  - [ ] Cleaning: missing data handling + duplicates
  - [ ] Compute acceptance rate + outcome distribution
  - [ ] Compute GPA/SAT/Extracurricular histograms
  - [ ] Export `summary.csv` and `summary.html`
  - [ ] Rule-based narrative insight generator
  - [ ] Skip gender/program sections when columns are not present (dataset has only GPA/SAT/Extracurricular/Admission_Status)

## Step 3: GitHub-ready project files
- [ ] Create `README.md` with run instructions + description of outputs
- [ ] Create `requirements.txt`
- [ ] Create `.gitignore`

## Step 4: Runtime validation
- [ ] Run the script and ensure outputs are created under `Project/outputs`

## Step 5: GitHub push (user-side)
- [ ] Initialize git, commit, and push (commands provided after validation)

