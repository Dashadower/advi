import jax.numpy as np
from jax import random
from jax import value_and_grad


class NormalMeanField:
    def __init__(self, num_params, rng_seed):
        self.num_params = num_params
        self.mu = np.zeros(num_params)
        self.omega = np.zeros(num_params)
        self.zeta = np.zeros((50, num_params))
        self.rng_key = random.PRNGKey(rng_seed)

    def sample(self, n_draws=1):
        """
        :param n_draws:
        :return: array of dimension(n_draws, self.num_params)
        """
        self.rng_key = random.split(self.rng_key)[1]  # need to recreate key after every use!!
        eta = random.normal(self.rng_key, (n_draws, self.omega.shape[0]))
        self.zeta = np.tile(self.mu, (n_draws, 1)) + np.exp(self.omega) * eta
        return self.zeta

    def entropy(self):
        return 0.5 * self.num_params * (1.0 + np.log(2 * np.pi)) + np.sum(self.omega)

    """def calc_elbo_and_grad(self, model, n_samples):
        def elbo(mu, log_sigma, _model, approx, _n_samples):
            samples = approx.sample(mu=mu, log_sigma=log_sigma, n_draws=_n_samples)  # batch process all draws
            lp = _model.lp(samples)

            return lp / _n_samples + approx.entropy()

        return value_and_grad(elbo, (0, 1))(self.mu, self.log_sigma, model, self, n_samples)  # grad wrt self.mu, self.sigma"""

    def elbo(self, model, n_samples):
        lp = 0
        self.sample(n_draws=n_samples)
        """for x in range(n_samples):
            lp += model.lp(samples[x, :])"""
        lp += model.lp_sum(self.zeta)

        return lp / n_samples + self.entropy()

    def elbo_grad(self, model, n_samples):
        eta = (self.zeta - np.tile(self.mu, (n_samples, 1))) / np.tile(np.exp(self.omega), (n_samples, 1))
        lp_grad_temp = np.zeros(self.num_params, float)
        omega_grad = np.zeros(self.num_params, float)
        for x in range(n_samples):
            _ , lp_grad = value_and_grad(model.unconstrain_lp)(self.zeta[x, :])
            lp_grad_temp += lp_grad
            omega_grad += lp_grad * eta[x, :] * np.exp(self.omega[x])
        mu_grad = lp_grad_temp / n_samples
        omega_grad /= n_samples
        omega_grad += 1

        return mu_grad, omega_grad


if __name__ == '__main__':
    mf = NormalMeanField(10, 25123)
    print(mf.sample(5)[1, :])
    #print(mf.sample(5))