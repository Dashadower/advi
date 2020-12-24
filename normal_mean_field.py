import jax.numpy as np
from jax import random
from jax import value_and_grad


class NormalMeanField:
    def __init__(self, num_params, rng_seed):
        self.num_params = num_params
        self.mu = np.zeros(num_params)
        self.log_sigma = np.zeros(num_params)
        self.rng_key = random.PRNGKey(rng_seed)

    def sample(self, mu, log_sigma, n_draws=1):
        """
        :param mu:
        :param log_sigma:
        :param n_draws:
        :return: array of dimension(n_draws, self.num_params)
        """
        mu = mu if isinstance(mu, np.ndarray) else self.mu
        log_sigma = log_sigma if isinstance(log_sigma, np.ndarray) else self.log_sigma

        self.rng_key = random.split(self.rng_key)[1]  # need to recreate key after every use!!
        return mu + self.log_sigma_to_sigma_sq(log_sigma) * random.normal(self.rng_key, (n_draws, log_sigma.shape[0]))

    def log_sigma_to_sigma_sq(self, log_sigma):
        return np.exp(log_sigma)

    def entropy(self):
        return 0.5 * self.num_params * (1.0 + np.log(2 * np.pi)) + np.sum(self.log_sigma)

    def calc_elbo_and_grad(self, model, n_samples):
        def elbo(mu, log_sigma, _model, approx, _n_samples):
            samples = approx.sample(mu=mu, log_sigma=log_sigma, n_draws=_n_samples)  # batch process all draws
            lp = _model.lp(samples)

            return lp / _n_samples + approx.entropy()

        return value_and_grad(elbo, (0, 1))(self.mu, self.log_sigma, model, self, n_samples)  # grad wrt self.mu, self.sigma

if __name__ == '__main__':
    mf = NormalMeanField(10, 25123)
    print(mf.sample(mf.mu, mf.log_sigma,n_draws=1))
    print(mf.sample(mf.mu, mf.log_sigma,n_draws=1))