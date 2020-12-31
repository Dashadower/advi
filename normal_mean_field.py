import jax.numpy as np
from jax import random
from jax import value_and_grad, vmap


class NormalMeanField:
    def __init__(self, num_params, rng_seed):
        self.num_params = num_params
        self.mu = np.zeros(num_params)  # mean value
        self.omega = np.zeros(num_params)  # sqrt(sigma)
        self.rng_key = random.PRNGKey(rng_seed)

    def sample(self, n_draws=1):
        """
        Return samples from the mean field distribution
        :param n_draws: number of sample draws
        :return: array of dimension(n_draws, self.num_params)
        """
        self.rng_key = random.split(self.rng_key)[1]  # need to recreate key after every use!!
        eta = random.normal(self.rng_key, (n_draws, self.num_params))
        return np.tile(self.mu, (n_draws, 1)) + np.exp(self.omega) * eta

    def entropy(self):
        return 0.5 * self.num_params * (1.0 + np.log(2 * np.pi)) + np.sum(self.omega)

    def elbo_and_grad(self, model, n_samples):
        """

        :param model: Model object
        :param n_samples: number of samples to approximate log prob of Model
        :return: elbo,mu_grad,omega_grad where elbo is the elbo value, mu_grad gradient array wrt mu, omega_grad
        gradient array wrt omega
        """
        zeta = self.sample(n_samples)
        eta = (zeta - np.tile(self.mu, (n_samples, 1))) / np.tile(np.exp(self.omega), (n_samples, 1))
        elbo, lp_grad = vmap(value_and_grad(model.unconstrain_lp), in_axes=0, out_axes=0)(zeta)

        mu_grad = np.sum(lp_grad, axis=0)
        omega_grad = np.sum(lp_grad * eta * np.exp(self.omega), axis=0)

        mu_grad /= n_samples
        omega_grad /= n_samples
        omega_grad += 1

        return (np.sum(elbo)/n_samples + self.entropy()), mu_grad, omega_grad


if __name__ == '__main__':
    mf = NormalMeanField(10, 25123)
    print(mf.sample(5)[1, :])
    #print(mf.sample(5))