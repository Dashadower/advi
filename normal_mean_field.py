import jax.numpy as np
from jax.scipy.stats import multivariate_normal
from jax.random import normal as randn
from jax.random import PRNGKey
from jax import value_and_grad


class NormalMeanField:
    def __init__(self, num_params):
        self.num_params = num_params
        self.mu = np.zeros(num_params)
        self.log_sigma = np.ones(num_params)
        self.rng_key = PRNGKey(0)

    def sample(self, mu=None, log_sigma=None):
        mu = mu if isinstance(mu, np.ndarray) else self.mu
        log_sigma = log_sigma if isinstance(log_sigma, np.ndarray) else self.log_sigma
        return mu + self.log_sigma_to_sigma_sq(log_sigma) * randn(self.rng_key, log_sigma.shape)

    def lp(self, mu, log_sigma, x):
        return multivariate_normal.logpdf(x, mu, np.diag(self.log_sigma_to_sigma_sq(log_sigma)))

    def log_sigma_to_sigma_sq(self, log_sigma):
        return np.exp(2 * log_sigma)

    def entropy(self):
        return 0.5 * self.num_params + (1.0 + np.log(2 * np.pi)) + np.sum(self.log_sigma)

    def calc_elbo_and_grad(self, model, n_samples):
        def elbo(mu, log_sigma, _model, approx, _n_samples):
            elbo = 0.0
            for n in range(_n_samples):
                sample = approx.sample(mu=mu, log_sigma=log_sigma)
                lp = _model.lp(_model.convert_vector_to_param_dict(sample))
                elbo += lp

            return elbo / n_samples + approx.entropy()

        return value_and_grad(elbo, (0, 1))(self.mu, self.log_sigma, model, self, n_samples)  # grad wrt self.mu, self.sigma

