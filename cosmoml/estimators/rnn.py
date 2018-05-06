import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell

__all__ = ['RNNClassifier']


def _rnn_model_fn(features, labels, hidden_units, optimizer, activation_fn,
                  normalizer_fn, in_dropout, out_dropout, is_bidirectional,
                  is_attended,
                  mode):
  # Check for training mode
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  if not is_training:
    in_dropout = 1.0
    out_dropout = 1.0

  # Extracts the features
  input_layer = features['ts']
  length_layer = features['length']

  # Builds a recurrent neural network
  def cell(num_hidden):
    return DropoutWrapper(LSTMCell(num_hidden),
                          input_keep_prob=in_dropout,
                          output_keep_prob=out_dropout)

  # Defines the LSTM network
  network = MultiRNNCell([cell(n) for n in hidden_units])

  # Apply it to the input data
  if is_bidirectional:
    rnn_outputs, final_state = \
      tf.nn.bidirectional_dynamic_rnn(cell_fw=network,
                                      cell_bw=network,
                                      inputs=input_layer,
                                      sequence_length=length_layer,
                                      scope='rnn',
                                      dtype=tf.float32)

    # concat 2 directions' outputs together horizontally
    rnn_outputs = tf.concat([rnn_outputs[0], rnn_outputs[1]],
                            axis=2)

  else:
    rnn_outputs, final_state = tf.nn.dynamic_rnn(network, input_layer,
                                                 sequence_length=length_layer,
                                                 dtype=tf.float32,
                                                 scope='rnn')
  # Attention
  if is_attended:
    batch_size, output_len, hidden_len = rnn_outputs.get_shape().as_list()

    attention = tf.matmul(rnn_outputs, rnn_outputs, transpose_b=True)  # B L L
    # for softmax only accepts 2D tensor
    attention = tf.reshape(attention, [-1, output_len])  # (B x L) x L

    attention_weights = tf.nn.softmax(attention)
    attention_weights = tf.reshape(attention_weights,
                                   [-1, output_len, output_len])  # B x L x L

    # now get the re-weighted outputs
    rnn_outputs = tf.matmul(attention_weights, rnn_outputs)  # B x L x H
    # attention_outputs = tf.matmul(attention_weights, rnn_outputs)  # B x L x H

    # # get the attended output now
    # last_rnn_output = attention_outputs[:, output_len-1, :] # B x 1 x H
    # last_rnn_output = tf.squeeze(last_rnn_output)  # B x H
    # last_rnn_output.set_shape([None, hidden_len])

  # sum all outputs B x L x H => B x H
  last_rnn_output = tf.reduce_sum(rnn_outputs, axis=1)

  # Applies a last MLP for building the final prediction
  logits = slim.fully_connected(last_rnn_output, 64, scope='fc/fc_1',
                                activation_fn=tf.nn.elu)

  logits = tf.reshape(
    slim.fully_connected(logits, 1, scope='fc/fc_3', activation_fn=None), (-1,))

  prob = tf.nn.sigmoid(logits)

  predictions = {'prob': prob, 'last_rnn': last_rnn_output, 'rnn': rnn_outputs}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      export_outputs={
                                        'prob': tf.estimator.export.PredictOutput(predictions),
                                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)})

  # Compute and register loss function
  loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32),
                                            logits=logits))
  tf.losses.add_loss(loss)
  total_loss = tf.losses.get_total_loss()

  train_op = None
  eval_metric_ops = None

  # Define optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      # TODO: put learning rate into params list
      train_op = optimizer(learning_rate=0.00002).\
        minimize(loss=total_loss, global_step=tf.train.get_global_step())

    # tensorboard
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
               in_dropout=1.0,
               out_dropout=1.0,
               model_dir=None,
               is_bidirectional=False,
               is_attended=False,
               config=None):
    """Initializes a `MDNEstimator` instance.
    """

    def _model_fn(features, labels, mode):
      return _rnn_model_fn(features, labels,
                           hidden_units,
                           optimizer,
                           activation_fn,
                           normalizer_fn,
                           in_dropout,
                           out_dropout,
                           is_bidirectional,
                           is_attended,
                           mode)

    super(self.__class__, self).__init__(model_fn=_model_fn,
                                         model_dir=model_dir,
                                         config=config)
