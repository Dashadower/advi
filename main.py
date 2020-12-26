import numpy as np
def vi(model, approx_model, iter_count, step_size, n_mc_samples):
    model.unconstrain_params() #TODO where would unconstrain/constrain module belong?
    for x in range(iter_count):
        elbo = approx_model.elbo(model, n_mc_samples)
        mu_grad, omega_grad = approx_model.elbo_grad(model, n_mc_samples)
        approx_model.mu += step_size * mu_grad
        approx_model.omega += step_size * omega_grad

        if x % 100 == 0:
            print(f"ITERATION {x}, ELBO VALUE: {elbo}")
            print(model.report_constrain_params(approx_model.mu))
            #model.pprint(model.convert_vector_to_param_dict(model.report_constrain_params(approx_model.mu)))

    print("-" * 20)
    print("ALL ITERATIONS FINISHED")
    print("FINAL RESULTS:")
    print("ELBO:", elbo)
    print("44:", model.report_constrain_params(approx_model.mu))
    #model.pprint(model.convert_vector_to_param_dict(model.report_constrain_params(approx_model.mu)))

    # sol from mcmc 7.93, 6.36, .39, 0, -.2, -.02, -.35, -.18, .34, .06

if __name__ == '__main__':
    from normal_mean_field import NormalMeanField
    from models import EightSchools
    import time
    start = time.time()
    step_size = 0.01
    n_mc_samples = 30
    iters = 3000
    init_arr = np.array([8.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model = EightSchools(init_arr)
    approx = NormalMeanField(model.param_count, 20201224)
    approx.mu = np.array([8.0, np.log(6.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    vi(model, approx, iters, step_size, n_mc_samples)
    print(f"total time: {time.time() - start} seconds")