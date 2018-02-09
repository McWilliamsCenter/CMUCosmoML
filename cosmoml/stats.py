import tensorflow as tf
import math
c = - 0.5 * math.log(2*math.pi)

def log_normal2(x, mean, log_var, eps=0.0):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as log variance rather than standard deviation, which ensures :math:`\sigma > 0`.

        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)

    Parameters
    ----------
    x :  tensor
        Values at which to evaluate pdf.
    mean :  tensor
        Mean of the Gaussian distribution.
    log_var :  tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.

    Returns
    -------
     tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    return c - log_var/2 - (x - mean)**2 / (2 * tf.exp(log_var) + eps)

def log_gm2(x, mean, log_var, weights, eps=0.0):
    """
    Computes the log likelihood of x under a Gaussian Mixture Model

    Parameters
    ----------
    x : tensor
        Sample to evaluater, of shape (batch_size, ndim).
    mean : tensor
        Means of the p Gaussians, of shape (batch_size, ndim, ngaussians).
    log_var : tensor
        Log variances of the p Gaussians, of shape (batch_size, ndim, ngaussians).
    weigts : tensor
        Weights of the Gaussian Mixture p, of shape (batch_size, ndim, ngaussians).
    eps : float
        Small number added to denominator to avoid NaNs.

    Returns
    -------
    tensor, shape (batch_size, ndim)
    """
    log_norms = log_normal2(tf.expand_dims(x,-1), mean, log_var,eps=eps)
    max_log_norms = tf.reduce_max(log_norms, axis=-1)
    norms = tf.exp(log_norms - tf.expand_dims(max_log_norms, -1))
    return max_log_norms + tf.log(tf.reduce_sum( weights * norms , axis = -1))
