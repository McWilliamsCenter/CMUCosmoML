from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .networks import deep_mlp
from .stats import log_gm2

__all__ = ['MDNEstimator']

def _mdn_model_fn(features, labels, hidden_units, n_mixture,
               feature_columns, label_columns, optimizer, activation_fn, normalizer_fn, dropout, mode):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    label_dimension=len(label_columns)

    # Extracts the features
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)

    # Builds the neural network
    net = deep_mlp(input_layer, hidden_units, activation_fn, normalizer_fn,
                   dropout, is_training)

    # Create mixture components
    out_mu = slim.fully_connected(net, label_dimension*n_mixture , activation_fn=None)
    out_mu = tf.reshape(out_mu, (-1, label_dimension, n_mixture))

    out_logvar = slim.fully_connected(net, label_dimension*n_mixture , activation_fn=None)
    out_logvar = tf.clip_by_value(out_logvar, -10, 10) # For stability
    out_logvar = tf.reshape(out_logvar, (-1, label_dimension, n_mixture))

    out_p = slim.fully_connected(net, label_dimension*n_mixture , activation_fn=None)
    out_p = tf.nn.softmax(tf.reshape(out_p, (-1, label_dimension, n_mixture)))

    predictions = {'mu': out_mu, 'sigma': tf.exp(0.5*out_logvar), 'p':out_p}

    if mode == tf.estimator.ModeKeys.PREDICT:

        # Add functionality to sample directly from the model
        random_indices = tf.multinomial(tf.log(tf.reshape(out_p, (-1, n_mixture))), 1)
        random_indices = tf.reshape(random_indices, (-1,))

        # Create mask to select these indices
        mask = tf.one_hot(random_indices, n_mixture, on_value = True, off_value = False, dtype = tf.bool)
        # Select the Gaussians used in the sampling
        mu = tf.boolean_mask(tf.reshape(out_mu, (-1, n_mixture)), mask)
        logvar =  tf.boolean_mask(tf.reshape(out_logvar, (-1, n_mixture)), mask)

        # Sample from Gaussian
        y = mu + tf.exp(0.5 * logvar) * tf.random_normal(tf.shape(mu))
        y = tf.reshape(y, (-1, label_dimension))

        samples = {}
        for i,k in enumerate(label_columns):
            samples[k.name] = y[:,-i-1]

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'pdf': tf.estimator.export.PredictOutput(predictions),
                                                          'samples': tf.estimator.export.PredictOutput(samples),
                                                          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(samples)})

    label_layer = tf.feature_column.input_layer(features=features, feature_columns=label_columns)

    # Compute and register loss function
    loss = - tf.reduce_mean(log_gm2(label_layer, out_mu, out_logvar, out_p))
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
               optimizer=tf.train.AdamOptimizer,
               activation_fn=tf.nn.relu,
               normalizer_fn=slim.batch_norm,
               dropout=None,
               model_dir=None,
               config=None):
        """Initializes a `MDNEstimator` instance.
        """

        def _model_fn(features, labels, mode):
            return _mdn_model_fn(features, labels, hidden_units, n_mixture,
               feature_columns, label_columns, optimizer, activation_fn, normalizer_fn, dropout, mode)

        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)
