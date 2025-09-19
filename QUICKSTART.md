Quickstart
----------
```python
import polars as pl
from mice_imputer import MiceImputer

df = pl.DataFrame({
  "age": [23, None, 45, 36, None],
  "income": [52000, 61000, None, 45000, 70000],
  "education": ["college", None, "graduate", "high_school", "college"],
  "region": ["MW", "NE", "S", "W", "MW"],
})
meta = {
  "age": {"type": "continuous", "pmm": True, "pmm_k": 5, "range": (0, 120)},
  "income": {"type": "continuous", "pmm": True, "pmm_k": 5, "log_transform": True},
  "education": {"type": "categorical", "pmm": False,
                "order": ["elementary", "high_school", "college", "graduate"]},
  "region": {"type": "categorical", "pmm": False}
}
imputer = MiceImputer(metadata=meta, n_iterations=5, random_state=42)
df_imputed = imputer.fit_transform(df)
print(df_imputed)
```

Notes
-----
- This is a pragmatic, minimal implementation intended for research workflows.
- We prioritize clarity and reproducibility over maximum speed.
- PMM is only used for continuous targets.

Dependencies
------------
polars, numpy, scikit-learn, lightgbm, joblib, pyarrow (for parquet)

Design choices
--------------
- Helper functions are defined before the class (topologically sorted).
- snake_case naming throughout and 2-space indentation.
- Reproducibility via numpy.random.Generator seeded from random_state.
- Safe log transform uses log1p/expm1 so non-positive values work.
- Imputation order by descending missingness to stabilize early passes.
- Simple convergence check based on L2 change across imputed cells.
