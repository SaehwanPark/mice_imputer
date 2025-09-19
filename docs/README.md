# MICE-Like Missing Values Imputer: Project Specification

## Overview

A high-performance, flexible missing values imputation library implementing Multiple Imputation by Chained Equations (MICE) using LightGBM predictions and Predictive Mean Matching (PMM). Built on modern Python data stack for production use.

## Core Architecture

### Backend Technologies
- **Dataframe**: Polars (primary), NumPy (interoperability layer)
- **ML Prediction**: LightGBM (gradient boosting trees)
- **PMM Implementation**: Scikit-learn (kNN search)
- **Output Format**: Parquet files via Polars

### Design Principles
- Scikit-learn style API (`fit()`, `transform()`, `fit_transform()`)
- Type hints throughout codebase
- Snake_case naming convention
- Production-ready: serializable models, efficient memory usage

## Key Features

### 1. Flexible Column Selection
```python
# Specify which columns to impute
columns_to_impute = ['age', 'income', 'education']

# Custom predictor sets per target column
predictor_mapping = {
    'age': ['income', 'education', 'region'],
    'income': ['age', 'education'],
    'education': ['age', 'income']
}
```

### 2. Sophisticated Categorical Handling
- **One-hot encoded variables**: Automatic reconstruction → imputation → re-encoding
- **Validation**: Binary constraint checking, one-hot rule compliance
- **Mixed-type predictors**: Native support in LightGBM

### 3. Metadata-Driven Configuration
```python
metadata = {
    'age': {
        'type': 'continuous',
        'pmm': True,
        'pmm_k': 5,
        'range': (0, 120)
    },
    ('gender_male', 'gender_female'): {
        'type': 'onehot',
        'original_name': 'gender',
        'pmm': False,
        'validate_exclusive': True
    },
    'education': {
        'type': 'ordinal',
        'order': ['elementary', 'high_school', 'college', 'graduate'],
        'pmm': False
    },
    'income': {
        'type': 'continuous',
        'pmm': True,
        'pmm_k': 10,
        'log_transform': True
    }
}
```

### 4. Model Persistence
- Save fitted imputers for future use
- Apply to new datasets with identical structure
- Version compatibility tracking

## Technical Implementation

### Class Structure

```python
class MiceImputer:
    def __init__(self, 
                 metadata: Dict = None,
                 n_iterations: int = 10,
                 lgbm_params: Dict = None,
                 random_state: int = None,
                 convergence_threshold: float = 1e-3):
        """
        Parameters:
        -----------
        metadata : dict
            Column metadata for imputation configuration
        n_iterations : int
            Maximum MICE iterations
        lgbm_params : dict
            LightGBM parameters
        random_state : int
            Random seed for reproducibility
        convergence_threshold : float
            Threshold for iteration convergence
        """
    
    def fit(self, X: pl.DataFrame, 
            columns_to_impute: List[str] = None,
            predictor_mapping: Dict[str, List[str]] = None) -> 'MiceImputer':
        """Fit MICE imputer on training data"""
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply imputation to new data"""
    
    def fit_transform(self, X: pl.DataFrame, **fit_params) -> pl.DataFrame:
        """Fit and transform in single call"""
    
    def save(self, filepath: str) -> None:
        """Serialize fitted imputer"""
    
    @classmethod
    def load(cls, filepath: str) -> 'MiceImputer':
        """Load serialized imputer"""
```

### Core Methods

```python
def _preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    - Validate one-hot groups
    - Reconstruct categorical variables
    - Apply transformations (log, etc.)
    """

def _fit_iteration(self, data: pl.DataFrame, iteration: int) -> pl.DataFrame:
    """
    Single MICE iteration:
    1. For each column to impute:
       - Extract predictors and target
       - Train LightGBM model
       - Generate predictions
       - Apply PMM or direct sampling
       - Update missing values
    """

def _train_column_model(self, target_col: str, predictor_cols: List[str], 
                       data: pl.DataFrame) -> lgb.LGBMRegressor:
    """Train LightGBM model for single column"""

def _pmm_sample(self, predictions: np.ndarray, observed_values: np.ndarray, 
               k: int = 5) -> np.ndarray:
    """
    Predictive Mean Matching:
    1. Find k nearest observed values to each prediction
    2. Sample randomly from donor pool
    """

def _sample_categorical(self, predictions: np.ndarray, 
                       categories: List[str]) -> np.ndarray:
    """Sample categorical values from model probabilities"""

def _check_convergence(self, current_data: pl.DataFrame, 
                      previous_data: pl.DataFrame) -> bool:
    """Monitor convergence across iterations"""

def _postprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    - Re-encode categorical variables to one-hot
    - Apply inverse transformations
    - Validate constraints
    """
```

### Validation Framework

```python
def _validate_onehot_group(self, columns: Tuple[str], df: pl.DataFrame) -> None:
    """
    Validate one-hot encoded variable group:
    1. All columns are binary (0/1 only)
    2. Each row has exactly one 1 (ignoring nulls)
    3. Warn on violations, error on severe issues
    """

def _validate_metadata(self, df: pl.DataFrame) -> None:
    """
    Validate metadata consistency:
    1. Referenced columns exist in data
    2. Type specifications match data types
    3. Constraint ranges are reasonable
    """

def _validate_predictor_mapping(self, mapping: Dict) -> None:
    """
    Validate predictor specifications:
    1. No circular dependencies
    2. Predictor columns exist
    3. Target columns are in imputation list
    """
```

## Default Behavior & Smart Inference

### Automatic Type Detection
```python
def _infer_column_metadata(self, col: str, data: pl.Series) -> Dict:
    """
    Smart defaults based on data characteristics:
    - Binary (2 unique values in {0,1}) → pmm=False
    - Low cardinality (<10 unique) → categorical, pmm=False  
    - High cardinality continuous → pmm=True, pmm_k=5
    - Detect potential one-hot groups by column naming patterns
    """
```

### Convergence Monitoring
- Track imputed value stability across iterations
- Early stopping when changes below threshold
- Maximum iteration limit as fallback

### Memory Optimization
- Polars lazy evaluation where possible
- Efficient numpy arrays for PMM calculations
- Minimal data copying during iterations

## Error Handling & Edge Cases

### Data Quality Issues
- Missing predictor columns: Warning + automatic exclusion
- All-missing columns: Skip imputation, preserve as missing
- Single-value columns: Warning + mode imputation

### One-Hot Violations
- Multiple 1's in row: Warning + use most confident prediction
- All 0's in row: Treat as missing categorical value
- Inconsistent group sizes: Error with clear diagnostic

### Model Training Failures
- Insufficient training data: Fallback to mean/mode imputation
- LightGBM convergence issues: Reduce complexity, retry
- PMM donor shortage: Expand search radius dynamically

## Output Specifications

### Completed Dataset
- Same structure as input (column order, types)
- No missing values in imputed columns
- Polars DataFrame format
- Optional parquet export with metadata preservation

### Imputation Diagnostics
```python
imputer.diagnostics_ = {
    'iterations_run': 8,
    'converged': True,
    'column_metrics': {
        'age': {'r2_score': 0.85, 'pmm_used': True},
        'income': {'r2_score': 0.72, 'pmm_used': True},
        'education': {'accuracy': 0.91, 'pmm_used': False}
    },
    'warnings': ['Column X had low prediction accuracy']
}
```

## Dependencies

### Core Requirements
```toml
# Core data/ML stack
polars = ">=0.20.0"
numpy = ">=1.24.0"
lightgbm = ">=4.0.0"
scikit-learn = ">=1.3.0"

# Serialization/IO
joblib = ">=1.3.0"
pyarrow = ">=12.0.0"  # for parquet support

# Optional enhancements
pandas = ">=2.0.0"  # for compatibility layer
```

### Development Requirements
```toml
pytest = ">=7.0.0"
pytest-cov = ">=4.0.0"
black = ">=23.0.0"
mypy = ">=1.5.0"
```

## Testing Strategy

### Unit Tests
- Individual method validation
- Edge case handling
- Metadata parsing correctness
- One-hot reconstruction accuracy

### Integration Tests  
- End-to-end imputation workflows
- Cross-validation on benchmark datasets
- Performance regression tests
- Memory usage profiling

### Benchmark Datasets
- Titanic (mixed types, moderate missingness)
- Adult Income (categorical heavy)
- Boston Housing (continuous focus)
- Synthetic data with controlled missingness patterns

This specification provides a comprehensive foundation for implementation while maintaining flexibility for future enhancements like temporal dependencies and advanced constraint handling.