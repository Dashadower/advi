def vi(model, approx_model, iter_count, eta, n_mc_samples):
    for x in range(iter_count):
        elbo = approx_model.elbo(model, n_mc_samples)
        mu_grad, sigma_grad = approx_model.elbo_grad(model, n_mc_samples)
        approx_model.mu += eta * mu_grad
        approx_model.log_sigma += eta * sigma_grad

        if x % 100 == 0:
            print(f"ITERATION {x}, ELBO VALUE: {elbo}")
            model.pprint(model.convert_vector_to_param_dict(model.constrain_params(approx.mu)))
            print(mu_grad)
            print(sigma_grad)

    print("-" * 20)
    print("ALL ITERATIONS FINISHED")
    print("FINAL RESULTS:")
    print("ELBO:", elbo)
    model.pprint(model.convert_vector_to_param_dict(model.constrain_params(approx.mu)))


if __name__ == '__main__':
    from normal_mean_field import NormalMeanField
    from models import EightSchools
    import time
    start = time.time()
    eta = 0.01
    n_mc_samples = 50
    iters = 2000
    model = EightSchools()
    approx = NormalMeanField(model.param_count, 20201224)
    approx.mu = model.constrain_params(model.convert_param_dict_to_vector(
        {
            "mu": 3.0,
            "tau": 2.0,
            "theta_trans": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        }
    ))
    vi(model, approx, iters, eta, n_mc_samples)
    print(f"total time: {time.time() - start} seconds")
