import jax.numpy as np
from jax.scipy.stats.norm import logpdf
from jax.ops import index_update

class Model:
    def __init__(self, param_count, param_index_dict):
        """
        :param param_count: number of parameters to approximate
        :param param_index_dict: dict containing indexes for each parameter
        """
        self.param_count = param_count
        self.param_index_dict = param_index_dict

    def lp(self, params):
        """
        Return the log density of the model given params
        :param params: dict of params
        :return: lp value
        """
        return NotImplementedError

    def unconstrain_params(self, param_arr):
        """
        Return the unconstrained parameters.
        Convert from constrained to unconstrained
        :return: array of unconstrained parameters
        """
        return NotImplementedError

    def constrain_params(self, param_arr):
        """
        Given unconstrained param arr, return arr after constraining params
        :param param_arr: unconstrained param array
        :return: array of constrained parameters
        """
        return NotImplementedError

    def convert_vector_to_param_dict(self, vector):
        """
        Given vector, return a dict mapping each values to respective parameters
        :param vector: 1d np array of parameters
        :return: dict of parameters defined by self.param_index_dict
        """
        ret_dict = {}
        for key, val in self.param_index_dict.items():
            if isinstance(val, list):
                ret_dict[key] = []
                for idx in val:
                    ret_dict[key].append(vector[idx])
            else:
                ret_dict[key] = vector[val]
        return ret_dict

    def convert_param_dict_to_vector(self, param_dict):
        """
        Given a param dict, convert to a 1d array
        :param param_dict: dict of parameters
        :return: jax.numpy.ndarray
        """
        ret_arr = np.zeros(self.param_count)
        for key, val in param_dict.items():
            if isinstance(val, list):
                for index, subval in enumerate(val):
                    #ret_arr[self.param_index_dict[key][index]] = subval
                    ret_arr = index_update(ret_arr, self.param_index_dict[key][index], subval)
            else:
                #ret_arr[self.param_index_dict[key]] = val
                ret_arr = index_update(ret_arr, self.param_index_dict[key], val)

        return ret_arr


class EightSchools(Model):
    def __init__(self):
        param_dict = {
            "mu": 0,
            "tau": 1,
            "theta_trans": [2,3,4,5,6,7,8,9],  # array of 8 floats
        }
        super().__init__(8 + 1 + 1, param_dict)
        self.school_count = 8
        self.effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
        self.sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    def lp(self, params):
        if isinstance(params, np.ndarray):
            lp = 0
            mu_index = self.param_index_dict["mu"]
            tau_index = self.param_index_dict["tau"]
            theta_indexes = self.param_index_dict["theta_trans"]
            theta = np.tile(params[:, mu_index], (self.school_count, 1)).T + \
                    params[:, theta_indexes] * np.tile(params[:, tau_index], (self.school_count, 1)).T
            # theta is dim (n_draws, n_schools)
            lp += np.sum(logpdf(params[:, theta_indexes], 0, 1))
            lp += np.sum(logpdf(np.tile(self.effects, (theta.shape[0], 1)), theta, np.tile(self.sigma, (theta.shape[0], 1))))
            return lp

    def unconstrain_params(self, param_arr):
        param_arr = param_arr.copy()
        #param_arr[self.param_index_dict["tau"]] = np.log(param_arr[self.param_index_dict["tau"]])  # (0, int) -> R
        index_update(param_arr, self.param_index_dict["tau"], np.log(param_arr[self.param_index_dict["tau"]]))
        return param_arr

    def constrain_params(self, param_arr):
        param_arr = param_arr.copy()
        #param_arr[self.param_index_dict["tau"]] = np.exp(self.param_index_dict["tau"])  # R -> (0, inf)
        index_update(param_arr, self.param_index_dict["tau"], np.exp(param_arr[self.param_index_dict["tau"]]))
        return param_arr

    def pprint(self, param_dict):
        out = [f"mu: {param_dict['mu']}", f"tau: {param_dict['tau']}"]
        for x in range(self.school_count):
            out.append(f"theta_trans[{x}]: {param_dict['theta_trans'][x]}")

        print("\n".join(out))


if __name__ == '__main__':
    from normal_mean_field import NormalMeanField
    import time

    def analytic_8schools_lp(mu, tau, theta_trans):
        effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)
        theta = theta_trans * tau + mu
        lp = 0
        lp += sum(logpdf(theta_trans, 0, 1))
        lp += sum(logpdf(effects, theta, sigma))
        return lp
    # check vectorized lp works
    n_draws = 10
    mf = NormalMeanField(10, int(time.time()))
    mf.mu = np.array([3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float) # lp should be -41.24325221126154
    #mf.log_sigma = np.tile(-20, mf.num_params)
    st = time.time()
    samples = mf.sample(mf.mu, mf.log_sigma, n_draws=n_draws)
    print(samples)
    print(EightSchools().lp(samples) / n_draws, time.time() - st)
    print(analytic_8schools_lp(mf.mu[0], mf.mu[1], mf.mu[2:]))
    """st = time.time()
    lp = 0
    for x in range(n_draws):
        lp += EightSchools().lp(np.expand_dims(samples[x], 0))
    print(lp/ n_draws, time.time() - st)"""
