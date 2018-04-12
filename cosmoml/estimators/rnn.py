import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper, LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper

__all__ = ['RNNClassifier']

def _rnn_model_fn(features, labels, hidden_units, optimizer, activation_fn,
                    =normalizer_fn, dropout, mode):

    # Check for training mode
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    if not is_training:
        dropout=1.0

    # Extracts the features
    input_layer = features['ts']
    length_layer = features['length']

    # Builds a recurrent neural network
    def cell(num_hidden):
        return DropoutWrapper(LSTMCell(num_hidden, name='LSTM'), output_keep_prob=dropout)

    # Defines the LSTM network
    network = MultiRNNCell([cell(n) for n in hidden_units])

    # Apply it to the input data
    rnn_outputs, final_state = tf.nn.dynamic_rnn(network, input_layer,
                                                 sequence_length=length_layer,
                                                 dtype=tf.float32,
                                                 scope='rnn')
    last_rnn_output = final_state[-1].h

    # Applies a last MLP for building the final prediction
    logits = slim.fully_connected(last_rnn_output, 128, scope='fc/fc_1',  activation_fn=tf.nn.elu)
    #logits = slim.fully_connected(logits, 128, scope='fc/fc_2',  activation_fn=tf.nn.elu)
    logits = tf.reshape(slim.fully_connected(logits, 1, scope='fc/fc_3',  activation_fn=None), (-1,))

    prob = tf.nn.sigmoid(logits)

    predictions = {'prob': prob, 'last_rnn':last_rnn_output, 'rnn':rnn_outputs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'prob': tf.estimator.export.PredictOutput(predictions),
                                                          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)})

    # Compute and register loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits))
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()

    train_op = None
    eval_metric_ops = None

    # Define optimizer
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer(learning_rate=0.00002).minimize(loss=total_loss,
                                        global_step=tf.train.get_global_step())
        tf.summary.scalar('loss', total_loss)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"log_p": loss}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


class RNNClassifier(tf.estimator.Estimator):
    """A classifier for time series data
    """

    def __init__(self,
               hidden_units,
               optimizer=tf.train.AdamOptimizer,
               activation_fn=tf.nn.relu,
               normalizer_fn=slim.batch_norm,
               dropout=None,
               model_dir=None,
               config=None):
        """Initializes a `MDNEstimator` instance.
        """

        def _model_fn(features, labels, mode):
            return _rnn_model_fn(features, labels, hidden_units, optimizer, activation_fn,
                              normalizer_fn, dropout, mode)

        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)
