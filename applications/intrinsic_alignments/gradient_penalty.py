import tensorflow.contrib.gan as tfgan
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables as contrib_variables_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.losses.python import losses_impl
from tensorflow.python.util import tf_inspect

# We need to define a specialised gradient penalty to handle the non trivial shape of our data
def custom_wasserstein_gradient_penalty(
    real_data,
    generated_data,
    generator_inputs,
    discriminator_fn,
    discriminator_scope,
    epsilon=1e-10,
    target=1.0,
    one_sided=False,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
  """The gradient penalty for the Wasserstein discriminator loss.
  See `Improved Training of Wasserstein GANs`
  (https://arxiv.org/abs/1704.00028) for more details.
  Args:
    real_data: Real data.
    generated_data: Output of the generator.
    generator_inputs: Exact argument to pass to the generator, which is used
      as optional conditioning to the discriminator.
    discriminator_fn: A discriminator function that conforms to TFGAN API.
    discriminator_scope: If not `None`, reuse discriminators from this scope.
    epsilon: A small positive number added for numerical stability when
      computing the gradient norm.
    target: Optional Python number or `Tensor` indicating the target value of
      gradient norm. Defaults to 1.0.
    one_sided: If `True`, penalty proposed in https://arxiv.org/abs/1709.08894
      is used. Defaults to `False`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `real_data` and `generated_data`, and must be broadcastable to
      them (i.e., all dimensions must be either `1`, or the same as the
      corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  Raises:
    ValueError: If the rank of data Tensors is unknown.
  """
  with ops.name_scope(scope, 'custom_wasserstein_gradient_penalty',
                      (real_data, generated_data)) as scope:
    real_data = ops.convert_to_tensor(real_data)
    generated_data = ops.convert_to_tensor(generated_data)
    if real_data.shape.ndims is None:
      raise ValueError('`real_data` can\'t have unknown rank.')
    if generated_data.shape.ndims is None:
      raise ValueError('`generated_data` can\'t have unknown rank.')


    # We need to add noise to the input with proper shape, this can be done by normalising the
    # pooling matrix and multiplying by noise
    # First, extract the conditionals
    W_1_0, W_2_0, W_3_0, xsp_0, pm_1_0,pm_2_0,pm_3_0, X, noise, Y = generator_inputs

    # Entries normalised to one
    pool = tf.SparseTensor(tf.cast(pm_1_0, tf.int64), tf.ones_like(pm_2_0), pm_3_0)

    batch_size = pm_3_0[0]
    alpha = random_ops.random_uniform(shape=(batch_size, 1))
    alpha = tf.sparse_tensor_dense_matmul(pool, alpha, adjoint_a=True)

    differences = generated_data - real_data
    interpolates = real_data + (alpha * differences)

    with ops.name_scope(None):  # Clear scope so update ops are added properly.
      # Reuse variables if variables already exists.
      with variable_scope.variable_scope(discriminator_scope, 'gpenalty_dscope',
                                         reuse=variable_scope.AUTO_REUSE):
        disc_interpolates = discriminator_fn(interpolates, generator_inputs)

    if isinstance(disc_interpolates, tuple):
      # ACGAN case: disc outputs more than one tensor
      disc_interpolates = disc_interpolates[0]

    # Let's assume this is remains separated by batches
    gradients = gradients_impl.gradients(disc_interpolates, interpolates)[0]

    # Summing square gradients over batches and output dimension
    gradient_squares = tf.reduce_sum(tf.sparse_tensor_dense_matmul(pool, math_ops.square(gradients)),axis=1)

    # Propagate shape information, if possible.
    if isinstance(batch_size, int):
      gradient_squares.set_shape([batch_size] + gradient_squares.shape.as_list()[1:])

    # For numerical stability, add epsilon to the sum before taking the square
    # root. Note tf.norm does not add epsilon.
    slopes = math_ops.sqrt(gradient_squares + epsilon)
    penalties = slopes / target - 1.0
    if one_sided:
      penalties = math_ops.maximum(0., penalties)
    penalties_squared = math_ops.square(penalties)
    penalty = losses.compute_weighted_loss(
        penalties_squared, weights, scope=scope,
        loss_collection=loss_collection, reduction=reduction)

    if add_summaries:
      summary.scalar('gradient_penalty_loss', penalty)

    return penalty

def _args_to_gan_model(loss_fn):
  """Converts a loss taking individual args to one taking a GANModel namedtuple.
  The new function has the same name as the original one.
  Args:
    loss_fn: A python function taking a `GANModel` object and returning a loss
      Tensor calculated from that object. The shape of the loss depends on
      `reduction`.
  Returns:
    A new function that takes a GANModel namedtuples and returns the same loss.
  """
  # Match arguments in `loss_fn` to elements of `namedtuple`.
  # TODO(joelshor): Properly handle `varargs` and `keywords`.
  argspec = tf_inspect.getargspec(loss_fn)
  defaults = argspec.defaults or []

  required_args = set(argspec.args[:-len(defaults)])
  args_with_defaults = argspec.args[-len(defaults):]
  default_args_dict = dict(zip(args_with_defaults, defaults))

  def new_loss_fn(gan_model, **kwargs):  # pylint:disable=missing-docstring
    def _asdict(namedtuple):
      """Returns a namedtuple as a dictionary.
      This is required because `_asdict()` in Python 3.x.x is broken in classes
      that inherit from `collections.namedtuple`. See
      https://bugs.python.org/issue24931 for more details.
      Args:
        namedtuple: An object that inherits from `collections.namedtuple`.
      Returns:
        A dictionary version of the tuple.
      """
      return {k: getattr(namedtuple, k) for k in namedtuple._fields}
    gan_model_dict = _asdict(gan_model)

    # Make sure non-tuple required args are supplied.
    args_from_tuple = set(argspec.args).intersection(set(gan_model._fields))
    required_args_not_from_tuple = required_args - args_from_tuple
    for arg in required_args_not_from_tuple:
      if arg not in kwargs:
        raise ValueError('`%s` must be supplied to %s loss function.' % (
            arg, loss_fn.__name__))

    # Make sure tuple args aren't also supplied as keyword args.
    ambiguous_args = set(gan_model._fields).intersection(set(kwargs.keys()))
    if ambiguous_args:
      raise ValueError(
          'The following args are present in both the tuple and keyword args '
          'for %s: %s' % (loss_fn.__name__, ambiguous_args))

    # Add required args to arg dictionary.
    required_args_from_tuple = required_args.intersection(args_from_tuple)
    for arg in required_args_from_tuple:
      assert arg not in kwargs
      kwargs[arg] = gan_model_dict[arg]

    # Add arguments that have defaults.
    for arg in default_args_dict:
      val_from_tuple = gan_model_dict[arg] if arg in gan_model_dict else None
      val_from_kwargs = kwargs[arg] if arg in kwargs else None
      assert not (val_from_tuple is not None and val_from_kwargs is not None)
      kwargs[arg] = (val_from_tuple if val_from_tuple is not None else
                     val_from_kwargs if val_from_kwargs is not None else
                     default_args_dict[arg])

    return loss_fn(**kwargs)

  new_docstring = """The gan_model version of %s.""" % loss_fn.__name__
  new_loss_fn.__docstring__ = new_docstring
  new_loss_fn.__name__ = loss_fn.__name__
  new_loss_fn.__module__ = loss_fn.__module__
  return new_loss_fn

my_gradient_penaly = _args_to_gan_model(custom_wasserstein_gradient_penalty)
