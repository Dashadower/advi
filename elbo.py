#import numpy as np
import jax.numpy as np
from jax import grad


def compute_elbo_mean_field(approx, model, n_samples):



def compute_elbo_mean_grad(approx, model, n_samples):
    def grad_wrapper(mu, sigma):
        return compute_elbo_mean_field(approx, model, n_samples)
    return grad(grad_wrapper(approx.mu, approx.sigma))
