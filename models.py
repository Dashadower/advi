import jax.numpy as np
from jax.scipy.stats.norm import logpdf


class Model:
    def __init__(self, param_count, constrained_params):
        """
        :param param_count: number of parameters to approximate
        :param constrained_params: dict with array of constrained parameters as values
        """
        self.param_count = param_count
        self.constrained_params = constrained_params

    def lp(self, params):
        """
        Return the log density of the model given params
        :param params: dict of params
        :return: lp value
        """
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
        :param param_dict: unconstrained param dict
        :return: None
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
    def __init__(self, initial_mu=None, initial_tau=None, initial_theta=None):
        param_dict = {
            "mu": initial_mu,
            "tau": initial_tau,
            "theta_trans": initial_theta,  # array of 8 floats
        }
        super().__init__(8 + 1 + 1, param_dict)
        self.school_count = 8
        self.effects = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
        self.sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    def lp(self, params):
        lp = 0
        param_dict = self.return_unconstrained_params() if not params else params
        theta = np.full(self.school_count, param_dict["mu"]) + \
                param_dict["theta_trans"] * np.full(self.school_count, param_dict["tau"])

        lp += np.sum(logpdf(param_dict["theta_trans"], 0, 1))
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
            "theta_trans": vector[2:],  # array of 8 floats
        }
        return param_dict

    def pprint(self):
        out = []
        out.append(f"mu: {self.constrained_params['mu']}")
        out.append(f"tau: {self.constrained_params['tau']}")
        for x in range(8):
            out.append(f"theta_trans[{x}]: {self.constrained_params['theta_trans'][x]}")

        print("\n".join(out))
