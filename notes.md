## Mean-field Normal
For a parameter vector theta_1, theta_2, ..., theta_n in the
true model p(theta), MFN models N(mu_1, sigma_1), ... N(mu_n, sigma_n)
within approximation model q(). All parameters within q are deemed
independent of each other.

the density of the Mean field would be the product of all the
normal distributions within itself.

## Calculating ELBO
in stan and viabel:

```elbo = E[p(x)] + q.entropy()```

entropy for random variables: 

integrate from -inf to inf
-(q(x) ln q(x)) dx

= E[- ln q(x)]   # differential entropy

ELBO equation:
`E[ln q(x, theta)] - E[ln q(theta)]`

therefore we can just add entropy of q since it's equivalent to `-E[ln q(theta)]`

question: in viabel code, elbo is implemented as
```python
if approx.supports_entropy:
    lower_bound = np.mean(self.model(samples)) + approx.entropy(var_param)
else:
    lower_bound = np.mean(self.model(samples) - approx.log_density(samples))
```

```python
def log_density(x):
    return mvn.logpdf(x, param_dict['mu'], np.diag(np.exp(2*param_dict['log_sigma'])))
```
E[log q(theta)] would be the same as approximating log(q(x)) using a finite number of samples

##SGA

Recall that in a multivariate function, its function point increases
the fastest along its gradient. Its contrapositive also holds; 
its values decreases the fastest in the direction of -delta

So the most basic form of stochastic gradient ascent increments the current
function value in the direction of its gradient multiplied by some
some "learning rate" eta, which can be fixed or adaptive.

Since the function is at a critical point when its gradient is zero,
unless we "overshoot", we are guaranteed to go towards the local maximum
every step.

In the case of ELBO maximization, we apply this to each mu and sigma, just the 
gradient is calculated separately for mu and sigma.

## monte-carlo grad calculation
define # of MC samples n

randomly sample n times given parameter mu, sigma

calculate singular gradient value at each sample

use the gradients' mean

but ofc, we can just use an autodiff library

