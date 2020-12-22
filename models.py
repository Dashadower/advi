import jax.numpy as np
from jax.scipy.stats.norm import logpdf


class Model:
    def __init__(self, param_count, constrained_params):
        self.param_count = param_count
        self.constrained_params = constrained_params

    def lp(self, x):
        return NotImplementedError

    def return_unconstrained_params(self):
        """
        Return the unconstrained parameters.
        Convert from constrained to unconstrained
        :return: dict of unconstrained parameters
        """
        return NotImplementedError

    def set_constrained_params(self, param_dict):
        """
        Given unconstrained param dict, set the parameters of the model after constraining
        :param param_dict:
        :return:
        """
        return NotImplementedError

    def convert_vector_to_param_dict(self, vector):
        """
        Given unconstrained vector of sampled vectors, return a dict mapping each values to respective parameters
        :param vector: 1d np array of parameters
        :return: dict of parameters recognizable by the model
        """
        return NotImplementedError


class EightSchools(Model):
    def __init__(self, initial_mu=None, initial_tau=None, initial_eta=None):
        param_dict = {
            "mu": initial_mu,
            "tau": initial_tau,
            "eta": initial_eta,  # array of 8 floats
        }
        super().__init__(8 + 1 + 1, param_dict)
        self.school_count = 8
        self.effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
        self.sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    def lp(self, params):
        lp = 0
        param_dict = self.return_unconstrained_params() if not params else params
        theta = np.full(self.school_count, param_dict["mu"]) + \
                np.full(self.school_count, param_dict["tau"]) * param_dict["eta"]

        lp += np.sum(logpdf(param_dict["eta"], 0, 1))
        lp += np.sum(logpdf(self.effects, theta, self.sigma))

        return lp

    def return_unconstrained_params(self):
        ret_dict = self.constrained_params.copy()
        ret_dict["tau"] = np.log(ret_dict["tau"])  # (0, int) -> R
        return ret_dict

    def set_constrained_params(self, param_dict):
        param_dict["tau"] = np.exp(param_dict["tau"])  # R -> (0, inf)
        self.constrained_params = param_dict

    def convert_vector_to_param_dict(self, vector):
        param_dict = {
            "mu": vector[0],
            "tau": vector[1],
            "eta": vector[2:],  # array of 8 floats
        }
        return param_dict

    def pprint(self):
        out = []
        out.append(f"mu: {self.constrained_params['mu']}")
        out.append(f"tau: {self.constrained_params['tau']}")
        for x in range(8):
            out.append(f"eta[{x}]: {self.constrained_params['eta'][x]}")

        print("\n".join(out))