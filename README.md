# MICE-Like Missing Value Imputer

A simple yet flexible **MICE-like imputer** for handling missing values in tabular datasets, built with **Polars**, **LightGBM**, and **Predictive Mean Matching (PMM)**.
Designed for research workflows where clarity, reproducibility, and diagnostics are as important as raw speed.

---

## ✨ Features

* Iterative **chained equations** imputation (MICE-like).
* Supports **continuous, categorical, and ordinal** variables.
* **Predictive Mean Matching (PMM)** for realistic draws on continuous variables.
* **LightGBM models** for flexible regression/classification.
* Handles **one-hot encoded groups** (round-trip reconstruction).
* Safe **log-transform option** for skewed variables.
* Optional **range enforcement** for continuous targets.
* Detailed **diagnostics**: per-iteration convergence, per-column metrics.
* **Save/Load** with `joblib`.

---

## 📦 Installation

We recommend using [**uv**](https://github.com/astral-sh/uv), a fast Python package manager.

```bash
uv pip install polars lightgbm scikit-learn numpy joblib pyarrow
```

Clone or copy the repository:

```bash
git clone https://github.com/your-org/mice-imputer.git
cd mice-imputer
```

---

## 🚀 Quickstart

```python
import polars as pl
from mice_imputer import MiceImputer

# Example dataset
df = pl.DataFrame({
  "age": [23, None, 45, 36, None],
  "income": [52000, 61000, None, 45000, 70000],
  "education": ["college", None, "graduate", "high_school", "college"],
  "region": ["MW", "NE", "S", "W", "MW"],
})

# Metadata describing columns
meta = {
  "age": {"type": "continuous", "pmm": True, "pmm_k": 5, "range": (0, 120)},
  "income": {"type": "continuous", "pmm": True, "pmm_k": 5, "log_transform": True},
  "education": {"type": "categorical", "order": ["elementary", "high_school", "college", "graduate"]},
  "region": {"type": "categorical"},
}

# Run imputer
imputer = MiceImputer(metadata=meta, n_iterations=5, random_state=42)
df_imputed = imputer.fit_transform(df)

print(df_imputed)
print(imputer.diagnostics_)
```

---

## ⚙️ Configuration

Each column can be annotated with metadata:

| Key             | Type  | Meaning                                                            |
| --------------- | ----- | ------------------------------------------------------------------ |
| `type`          | str   | `"continuous"`, `"categorical"`, or `"ordinal"`                    |
| `pmm`           | bool  | Whether to apply Predictive Mean Matching (continuous only)        |
| `pmm_k`         | int   | Number of donors in PMM                                            |
| `range`         | tuple | `(min, max)` bounds for continuous variables                       |
| `order`         | list  | Explicit order for ordinal variables                               |
| `log_transform` | bool  | Apply log1p/expm1 transformation (for skewed continuous variables) |

---

## 📊 Diagnostics

After fitting, inspect `imputer.diagnostics_`:

```python
{
  "iterations_run": 4,
  "converged": True,
  "per_iter_delta": [0.23, 0.07, 0.01, 0.003],
  "column_metrics": {
    "age": {"r2_score": 0.81, "pmm_used": True},
    "education": {"accuracy": 0.90, "log_loss": 0.31, "pmm_used": False}
  },
  "warnings": []
}
```

---

## 💾 Save / Load

```python
imputer.save("imputer.joblib")
loaded = MiceImputer.load("imputer.joblib")
df2 = loaded.transform(df_new)
```

---

## 🧪 Testing

This project includes a full pytest suite (`tests/test_mice_imputer.py`) covering both **normal** and **edge cases**.

Run tests:

```bash
uv pip install pytest
pytest -q
```

---

## 📂 Project Structure

```
mice-imputer/
├── mice_imputer.py          # Main imputer implementation
├── tests/
│   └── test_mice_imputer.py # Unit tests
└── README.md                # This file
```

---

## ⚖️ License

MIT License — free for academic and commercial use.
