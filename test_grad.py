from jax import random


from models import EightSchools
from normal_mean_field import NormalMeanField


def elbo(mu, log_sigma, _model, approx, _n_samples):
    sample = approx.sample(mu=mu, log_sigma=log_sigma, n_draws=_n_samples)
    lp = _model.lp(sample)

    return lp / _n_samples + approx.entropy()


def elbo_sigma(log_sigma, mu, _model, approx, _n_samples):
    sample = approx.sample(mu=mu, log_sigma=log_sigma, n_draws=_n_samples)
    lp = _model.lp(sample)

    return lp / _n_samples + approx.entropy()


if __name__ == '__main__':
    import time
    from scipy.optimize import approx_fprime
    test_random_key = random.PRNGKey(int(time.time()))

    model = EightSchools()
    approx = NormalMeanField(model.param_count, int(time.time()))
    test_mu = random.normal(test_random_key, (model.param_count,))
    test_sigma = random.normal(random.split(test_random_key)[1], (model.param_count,))

    approx.mu = test_mu
    approx.log_sigma = test_sigma
    approx.rng_key = test_random_key

    epsilon = 0.01

    print(test_mu)
    print(test_sigma)
    print("-" * 10)

    print("partials wrt mu:")
    print("fd:")
    print(approx_fprime(test_mu, elbo, epsilon, test_sigma, model, approx, 50))
    print("jax:")
    print(approx.calc_elbo_and_grad(model, 50)[1][0])

    print("\npartials wrt sigma:")
    print("fd:")
    print(approx_fprime(test_sigma, elbo_sigma, epsilon, test_mu, model, approx, 50))
    print("jax:")
    print(approx.calc_elbo_and_grad(model, 50)[1][1])
