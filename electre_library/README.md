# ELECTRE I DSS Package

A Python implementation of the ELECTRE I method for multi-criteria decision making, based on the steps outlined in a specific academic report.

## Installation

You can install this package locally using pip:

```bash
pip install -e . 
```
This installs the package in editable mode, which is useful during development. Make sure you have `numpy` installed or install it via:
```bash
pip install -r requirements.txt
```

## Usage

Import the solver and provide your data:

```python
import numpy as np
from electre_dss import ElectreSolver

# Example Data (from the report)
decision_matrix = np.array([
    [80, 70, 80, 70, 90], # HP1
    [80, 80, 70, 70, 90], # HP2
    [90, 70, 80, 70, 80]  # HP3
])

weights = np.array([5, 4, 3, 4, 2])

# C1, C4 are cost; C2, C3, C5 are benefit
criteria_types = ['cost', 'benefit', 'benefit', 'cost', 'benefit']

alternative_names = ['HP1', 'HP2', 'HP3']
criteria_names = ['C1', 'C2', 'C3', 'C4', 'C5']

# Initialize solver
solver = ElectreSolver(
    decision_matrix=decision_matrix,
    weights=weights,
    criteria_types=criteria_types,
    alternative_names=alternative_names,
    criteria_names=criteria_names
)

# Run analysis
solver.solve()

# Get results
ranking = solver.get_ranking()
print("Final Ranking:", ranking)

results = solver.get_intermediate_results()
# Access intermediate matrices like results['aggregate_dominance_matrix']
# print("\nAggregate Dominance Matrix:\n", results['aggregate_dominance_matrix'])
```

This package calculates the ranking based on the ELECTRE I method steps including normalization, weighting, concordance/discordance analysis, threshold calculation, and dominance matrix aggregation. 