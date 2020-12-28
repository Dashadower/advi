import numpy as onp
import time
def vi(model, approx_model, iter_count, step_size, n_mc_samples):
    assert iter_count > 0 and step_size > 0 and n_mc_samples > 0
    start_time = time.time()
    for x in range(iter_count):
        elbo, mu_grad, omega_grad = approx_model.elbo_and_grad(model, n_mc_samples)
        approx_model.mu += step_size * mu_grad
        approx_model.omega += step_size * omega_grad

        if x % 100 == 0:
            print(f"ITERATION {x}, ELBO VALUE: {elbo} IT/s: {x/(time.time()-start_time)}")
            print(list(model.constrain_param_array(approx_model.mu)))

    print("-" * 20)
    print(f"ALL ITERATIONS FINISHED. TOTAL TIME: {time.time() - start_time}")
    print("ELBO:", elbo)
    print("44:", list(model.constrain_param_array(approx_model.mu)))

    # sol from mcmc 7.93, 6.36, .39, 0, -.2, -.02, -.35, -.18, .34, .06


def adagrad_vi(model, approx_model, iter_count, step_size, n_mc_samples, tau=1):
    # use stan's adagrad algorithm
    start_time = time.time()
    mu_grad_accumulator = onp.zeros(model.param_count)
    omega_grad_accumulator = onp.zeros(model.param_count)
    assert iter_count > 0 and step_size > 0 and n_mc_samples > 0
    for x in range(iter_count):
        elbo, mu_grad, omega_grad = approx_model.elbo_and_grad(model, n_mc_samples)
        if x == 1:
            mu_grad_accumulator += onp.power(mu_grad_accumulator, 2)
            omega_grad_accumulator += onp.power(omega_grad, 2)
        else:
            mu_grad_accumulator *= 0.9
            mu_grad_accumulator += 0.1 * onp.power(mu_grad, 2)
            omega_grad_accumulator *= 0.9
            omega_grad_accumulator += 0.1 * onp.power(mu_grad, 2)

        step_size_scaled = step_size / onp.sqrt(x)
        approx_model.mu += step_size_scaled * mu_grad / (tau + onp.sqrt(mu_grad_accumulator))
        approx_model.omega += step_size_scaled * omega_grad / (tau + onp.sqrt(omega_grad_accumulator))

        if x % 100 == 0:
            print(f"ITERATION {x}, ELBO VALUE: {elbo} IT/s: {x/(time.time()-start_time)}")
            print(list(model.constrain_param_array(approx_model.mu)))

    print("-" * 20)
    print(f"ALL ITERATIONS FINISHED. TOTAL TIME: {time.time() - start_time}")
    print("ELBO:", elbo)
    print("44:", list(model.constrain_param_array(approx_model.mu)))

if __name__ == '__main__':
    from normal_mean_field import NormalMeanField
    from models import EightSchools
    import time
    step_size = 0.01
    n_mc_samples = 30
    iters = 10000
    init_arr = onp.array([8.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model = EightSchools(init_arr)
    approx = NormalMeanField(model.param_count, 20201224)
    #approx.mu = np.array([8.0, np.log(6.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    vi(model, approx, iters, step_size, n_mc_samples)