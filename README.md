# Simulating Memristive Neural Networks

Parts of the code are adapted from [this](https://github.com/joksas/nonideality-aware-mnn-training).

## Requirements

Python 3.9 and the packages listed in [requirements.txt](/requirements.txt).

## Reproducing results

To reproduce the results, run the following script:
```python
from mnn import simulations
from mnn.plotting import figures


simulations.discretisation.run()
figures.discretisation_boxplots()
```
