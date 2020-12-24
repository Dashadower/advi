import jax.numpy as np
from jax import random
from jax import value_and_grad


class NormalMeanField:
    def __init__(self, num_params, rng_seed):
        self.num_params = num_params
        self.mu = np.zeros(num_params)
        self.log_sigma = np.zeros(num_params)
        self.rng_key = random.PRNGKey(rng_seed)

    def sample(self, n_draws=1):
        """
        :param n_draws:
        :return: array of dimension(n_draws, self.num_params)
        """
        self.rng_key = random.split(self.rng_key)[1]  # need to recreate key after every use!!
        return self.mu + self.log_sigma_to_sigma_sq(self.log_sigma) * random.normal(self.rng_key, (n_draws, self.log_sigma.shape[0]))

    def log_sigma_to_sigma_sq(self, log_sigma):
        return np.exp(log_sigma)

    def entropy(self):
        return 0.5 * self.num_params * (1.0 + np.log(2 * np.pi)) + np.sum(self.log_sigma)

    """def calc_elbo_and_grad(self, model, n_samples):
        def elbo(mu, log_sigma, _model, approx, _n_samples):
            samples = approx.sample(mu=mu, log_sigma=log_sigma, n_draws=_n_samples)  # batch process all draws
            lp = _model.lp(samples)

            return lp / _n_samples + approx.entropy()

        return value_and_grad(elbo, (0, 1))(self.mu, self.log_sigma, model, self, n_samples)  # grad wrt self.mu, self.sigma"""

    def elbo(self, model, n_samples):
        samples = self.sample(n_samples)
        lp = 0
        """for x in range(n_samples):
            lp += model.lp(samples[x, :])"""
        lp += model.lp_sum(samples)

        return lp / n_samples + self.entropy()

    def elbo_grad(self, model, n_samples):
        zeta = self.sample(n_samples)
        eta = (zeta - np.tile(self.mu, (n_samples, 1))) / np.tile(self.log_sigma_to_sigma_sq(self.log_sigma), (n_samples, 1))
        lp_temp = 0
        lp_grad_temp = np.zeros(self.num_params, float)
        omega_grad = np.zeros(self.num_params, float)
        for x in range(n_samples):
            lp_zeta, lp_grad = value_and_grad(model.lp)(zeta[x, :])
            lp_temp += lp_zeta
            lp_grad_temp += lp_grad
            omega_grad += lp_grad * eta[x, :]

        mu_grad = lp_grad_temp / n_samples
        omega_grad /= n_samples
        omega_grad *= np.exp(self.log_sigma)
        omega_grad += 1

        return mu_grad, omega_grad


if __name__ == '__main__':
    mf = NormalMeanField(10, 25123)
    print(mf.sample(5)[1, :])
    #print(mf.sample(5))