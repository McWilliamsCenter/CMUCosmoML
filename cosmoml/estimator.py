from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as tfd

from .networks import deep_mlp

__all__ = ['MDNEstimator']

def _mdn_model_fn(features, labels, hidden_units, n_mixture, diag,
               feature_columns, label_columns, optimizer, activation_fn, normalizer_fn, dropout, mode):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    label_dimension=len(label_columns)

    # Extracts the features
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)

    # Builds the neural network
    net = deep_mlp(input_layer, hidden_units, activation_fn, normalizer_fn,
                   dropout, is_training)

    # Size of the covariance matrix
    if diag ==True:
        size_sigma = label_dimension
    else:
        size_sigma = (label_dimension *(label_dimension +1) // 2)

    # Create mixture components from network output
    out_mu = slim.fully_connected(net, label_dimension*n_mixture , activation_fn=None)
    out_mu = tf.reshape(out_mu, (-1, n_mixture, label_dimension))

    out_sigma = slim.fully_connected(net, size_sigma *n_mixture, activation_fn=None)
    out_sigma = tf.reshape(out_sigma, (-1,n_mixture, size_sigma))

    out_p = slim.fully_connected(net, n_mixture, activation_fn=None)
    out_p = tf.nn.softmax(tf.reshape(out_p, (-1, n_mixture)))

    if diag == True:
        sigma_mat = tf.nn.softplus(out_sigma)
        gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p),
                      components_distribution=tfd.MultivariateNormalDiag(loc=out_mu,
                                                                        scale_diag=sigma_mat))
    else:
        sigma_mat = tfd.matrix_diag_transform(tfd.fill_triangular(out_sigma), transform=tf.nn.softplus)
        gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=out_p),
                     components_distribution=tfd.MultivariateNormalTriL(loc=out_mu,
                                                                        scale_tril=sigma_mat))

    predictions = {'mu': out_mu, 'sigma': sigma_mat, 'p':out_p}

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Add functionality to sample directly from the model
        y = gmm.sample()

        samples = {}
        for i,k in enumerate(label_columns):
            samples[k.name] = y[:,-i-1]

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'pdf': tf.estimator.export.PredictOutput(predictions),
                                                          'samples': tf.estimator.export.PredictOutput(samples),
                                                          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(samples)})

    label_layer = tf.feature_column.input_layer(features=features, feature_columns=label_columns)

    # Compute and register loss function
    loss = - tf.reduce_mean(gmm.log_prob(label_layer),axis=0)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer(learning_rate=0.0002).minimize(loss=total_loss,
                                        global_step=tf.train.get_global_step())
        tf.summary.scalar('loss', loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = { "log_p": loss}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

class MDNEstimator(tf.estimator.Estimator):
    """An estimator for distribution estimation using Mixture Density Networks.
    """

    def __init__(self,
               feature_columns,
               label_columns,
               hidden_units,
               n_mixture,
               diagonal=True,
               optimizer=tf.train.AdamOptimizer,
               activation_fn=tf.nn.relu,
               normalizer_fn=slim.batch_norm,
               dropout=None,
               model_dir=None,
               config=None):
        """Initializes a `MDNEstimator` instance.
        """

        def _model_fn(features, labels, mode):
            return _mdn_model_fn(features, labels, hidden_units, n_mixture, diagonal,
               feature_columns, label_columns, optimizer, activation_fn, normalizer_fn, dropout, mode)

        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)
