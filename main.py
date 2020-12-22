#import numpy as np
import jax.numpy as np
def vi(model, approx_model, eta, n_mc_samples):
    keep_running = True
    for x in range(1000):
        elbo, grad = approx_model.calc_elbo_and_grad(model, n_mc_samples)  #elbo, mu_grad, sigma_grad
        mu_grad, sigma_grad = grad[0], grad[1]
        approx_model.mu = approx_model.mu + eta * mu_grad
        approx_model.log_sigma = approx_model.log_sigma  + eta * sigma_grad

        if x % 100 == 0:
            model.set_constrained_params(model.convert_vector_to_param_dict(approx_model.mu))
            print(x, elbo)
            model.pprint()



if __name__ == '__main__':
    from normal_mean_field import NormalMeanField
    from models import EightSchools
    eta = 0.025
    n_mc_samples = 50
    model = EightSchools()
    approx = NormalMeanField(model.param_count)
    model.set_constrained_params(model.convert_vector_to_param_dict(approx.sample()))
    vi(model, approx, eta, n_mc_samples)