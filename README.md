# Automatic Differentiation Variational Inference

A simple and crude implementation of ADVI with the help of Jax.

prerequisites: 
- Jax (the autograd package, not the champion!)

## A quick example

```python
from main import vi
from normal_mean_field import NormalMeanField
from models import EightSchools
eta = 0.025  # constant learning rate
n_mc_samples = 50  # number of MC samples for ELBO calculation
iters = 2000  # number of VI iterations
model = EightSchools()  # initialize eight school model
approx = NormalMeanField(model.param_count, 20201223)  # initialize mean field with a random seed
model.set_constrained_params(model.convert_vector_to_param_dict(approx.sample()))  # set the initial values of parameters with randomly sampled values
vi(model, approx, iters, eta, n_mc_samples)  # run vi. should print results to stdout
```
You can just run `main.py` and it does the same thing as the example.