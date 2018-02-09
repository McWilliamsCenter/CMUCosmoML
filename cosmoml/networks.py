from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import add_arg_scope

@add_arg_scope
def deep_mlp(inputs, hidden_units,
             activation_fn=tf.nn.relu,
             normalizer_fn=slim.batch_norm,
             dropout=None,
             is_training=True,
             reuse=None,
             outputs_collections=None, scope=None):
    """Defines a deep Multilayer perceptron
    """
    with tf.variable_scope(scope, 'mlp', [inputs],
                    reuse=reuse) as sc:

        net = inputs
        with slim.arg_scope([slim.fully_connected],
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    normalizer_params={'is_training': is_training}):

            for layer_id, num_hidden_units in enumerate(hidden_units):
                net = slim.fully_connected(inputs, num_hidden_units, scope='fc_%d'%layer_id)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                [net])
