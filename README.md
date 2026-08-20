# Data Science Internship — Titanic Survival Analysis

5-week internship project covering the full data science workflow: data cleaning,
exploratory analysis, visualization, statistical hypothesis testing, and machine
learning, all built on the Titanic passenger dataset.

## Structure

| Folder | Week | Focus |
|--------|------|-------|
| `week1/` | 1 | Data acquisition, cleaning, and exploratory analysis |
| `week2/` | 2 | Advanced data visualization and storytelling |
| `week3/` | 3 | Statistical hypothesis testing |
| `week4/` | 4 | Machine learning model development and evaluation |
| `week5/` | 5 | Comprehensive project synthesis and strategic recommendations |

`titanic_clean.csv` (in the repo root) is the cleaned dataset produced in Week 1
and reused as the input for Weeks 2-5.

## How to run

Each week's script is self-contained. From inside that week's folder:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
python3 <script_name>.py
```

Week 1 reads `titanic.csv` (raw data, included in `week1/`) and produces
`titanic_clean.csv`. Weeks 2-5 read `titanic_clean.csv` directly — copy it into
each week's folder before running, or run from the repo root.

## Key finding

Across all four analytical approaches used (visualization, hypothesis testing,
and machine learning), passenger class and sex were consistently the strongest
predictors of survival — first-class women survived at ~97%, third-class men at
~16%. Full write-ups with charts and interpretation are in the corresponding
Word report for each week (submitted separately per internship requirements).
