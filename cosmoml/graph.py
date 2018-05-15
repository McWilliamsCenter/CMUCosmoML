import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
from tensorflow.contrib.slim import add_arg_scope, model_variable

from math import sqrt

@add_arg_scope
def spatial_adjacency(features,
                              adjacency,
                              directions,
                              filter_size,
                              weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                              weights_regularizer=None,
                              biases_initializer=tf.zeros_initializer(),
                              biases_regularizer=None,
                              radial_scale=1.0,
			                  learn_scale=False,
                              reuse=None,
                              variables_collections=None,
                              outputs_collections=None,
                              trainable=True,
                              scope=None):
    """
    Weighted adjacency matrix based on directions, adapted from
    https://arxiv.org/pdf/1706.05206.pdf

    Generate a sparse adjacency matrix of shape (n, n, filter_size)
    """
    ss, n_features = features.get_shape()
    with tf.variable_scope(scope, 'spatial_adjacency',
                           [features, adjacency, directions], reuse=reuse) as sc:


        # Computes distances for each connection
        d = tf.matmul(features, directions)
        d = tf.gather(d, adjacency.indices[:,0]) - tf.gather(d, adjacency.indices[:,1])

        # Applying softmax along last dimension
        d = tf.nn.softmax(d)

        # Compute radial distance
        r = tf.gather(features, adjacency.indices[:,0]) - tf.gather(features, adjacency.indices[:,1])
        r2 = tf.reduce_sum(r**2, axis=1)

        s = model_variable('scale',dtype=tf.float32,
                            initializer=tf.constant(radial_scale, dtype=tf.float32),
                            trainable=learn_scale)

        # Apply distance scaling
        #dr = tf.exp(- 0.5 * r2/(s**2))
        d = d * tf.expand_dims(adjacency.values,axis=1)# tf.expand_dims(dr,axis=1)
        t = tf.SparseTensor(indices=adjacency.indices,
                            dense_shape=adjacency.dense_shape,
                            values=adjacency.values)

        # Renormalise the adjacency matrix
        t_inv = 1./ tf.sqrt(tf.sparse_reduce_sum(t, axis=1) + 1) # The one is for self connection
        t_inv = tf.gather(t_inv, adjacency.indices[:,0]) * tf.gather(t_inv, adjacency.indices[:,1])

        # Multiply d by the input adjacency values which should account for the
        # number of neighbours
        #d = d * tf.expand_dims(adjacency.values,axis=1)

        # Generating a list of sparse tensors as output
        q = []
        for i in range(0,filter_size):
            t = tf.SparseTensor(indices=adjacency.indices,
                                dense_shape=adjacency.dense_shape,
                                values=d[:,i] * t_inv)
            q.append(t)
        outputs = q

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def dynamic_adjacency(features,
                      adjacency,
                      filter_size,
                      weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                      weights_regularizer=None,
                      biases_initializer=tf.zeros_initializer(),
                      biases_regularizer=None,
                      reuse=None,
                      variables_collections=None,
                      outputs_collections=None,
                      trainable=True,
                      scope=None):
    """
    Dynamically weighted adjacency matrix, adapted from
    https://arxiv.org/pdf/1706.05206.pdf

    Generate a sparse adjacency matrix of shape (n, n, filter_size)
    """
    ss, n_features = features.get_shape()
    with tf.variable_scope(scope, 'dynamic_adjacency',
                           [features, adjacency], reuse=reuse) as sc:

        u = model_variable('dynamic_weights',
                            shape=[n_features, filter_size],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            trainable=True)

        c = model_variable('dynamic_bias',
                            shape=[filter_size],
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=True)

        # Computes distances for each connection
        d = tf.matmul(features, u)
        d = tf.gather(d, adjacency.indices[:,0]) - tf.gather(d, adjacency.indices[:,1])
        d = d + c
        # Applying softmax along last dimension
        d = tf.nn.softmax(d)

        # Multiply d by the input adjacency values which should account for the
        # number of neighbours
        d = d * tf.expand_dims(adjacency.values,axis=1)

        # Generating a list of sparse tensors as output
        q = []
        for i in range(0,filter_size):
                q.append( tf.SparseTensor(indices=adjacency.indices,
                                          dense_shape=adjacency.dense_shape,
                                          values=d[:,i]))
        outputs = q

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def graph_conv2(features,
                adjacency,
                num_outputs,
                one_hop=True,
                activation_fn=tf.nn.elu,
                weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    """
    Adds a fully connected layer.
    """
    ss, n_features = features.get_shape()

    if not isinstance(adjacency, list):
        adjacency = [adjacency]

    filter_size= len(adjacency)

    with tf.variable_scope(scope, 'graph_conv2', [features, adjacency], reuse=reuse) as sc:

        w0 = model_variable('weights_0',
                            shape=[n_features, num_outputs],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            trainable=True)

        if one_hop:
            w1 = model_variable('weights_1',
                                shape=[n_features, num_outputs, filter_size],
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=True)

        b = model_variable('bias',
                            shape=[num_outputs],
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=True)

        outputs = tf.matmul(features, w0)

        if one_hop:
            out = tf.tensordot(features, w1, axes=[[1], [0]])
            for i in range(0, filter_size):
                outputs += tf.sparse_tensor_dense_matmul(adjacency[i], out[:,:,i])

        outputs += b

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def ar_graph_conv2( features,
                adjacency,
                num_outputs_per_channel,
                n_channels,
                full_ar=False,
                one_hop=True,
                activation_fn=tf.nn.elu,
                weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    """
    Adds a fully connected layer.
    """
    ss, n_features = features.get_shape()
    num_outputs = num_outputs_per_channel * n_channels
    num_inputs_per_channel = tf.cast(n_features // n_channels, tf.int32)

    if not isinstance(adjacency, list):
        adjacency = [adjacency]

    filter_size= len(adjacency)

    with tf.variable_scope(scope, 'ar_graph_conv2', [features, adjacency], reuse=reuse) as sc:

        w0 = model_variable('weights_0',
                        shape=[n_features, num_outputs],
                        initializer=weights_initializer,
                        regularizer=weights_regularizer,
                        trainable=True)

        if one_hop:
            w1 = model_variable('weights_1',
                                shape=[n_features, num_outputs, filter_size],
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=True)

        b = model_variable('bias',
                            shape=[num_outputs],
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=True)


        # Build AR mask for features
        if full_ar:
            mask = tf.sequence_mask([ ( i / num_inputs_per_channel + 1 ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
            mask = tf.to_float( tf.logical_not(mask))
        else:
            mask = tf.sequence_mask([ ( i / num_inputs_per_channel ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
            mask = tf.to_float( tf.logical_not(mask))

        # If AR convolution, multiply the weights accordingly
        w0 = w0 * mask

        if one_hop:
            w1 = w1 * tf.expand_dims(mask, axis=2)

        # First, self-connection
        outputs = tf.matmul(features, w0)

        # If necessary, include neighbours
        if one_hop:
            out = tf.tensordot(features, w1, axes=[[1], [0]])
            for i in range(filter_size):
                outputs += tf.sparse_tensor_dense_matmul(adjacency[i], out[:,:,i])

        outputs += b

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def ar_graph_bottleneck2(inputs,
                         adjacency,
                         num_outputs_per_channel,
                         n_channels,
                         depth_bottleneck,
                         activation_fn=tf.nn.elu,
                         outputs_collections=None, scope=None):
    """
    Bottleneck residual unit variant
    """
    ss, n_features = inputs.get_shape()
    num_inputs_per_channel = tf.cast(n_features / n_channels, tf.int32)

    with tf.variable_scope(scope, 'ar_graph_bottleneck2', [inputs, adjacency]) as sc:

        #preact = layers.batch_norm(inputs, activation_fn=activation_fn, scope='preact')
        preact = inputs
        if num_inputs_per_channel == num_outputs_per_channel:
            shortcut = inputs
        else:
            shortcut = ar_graph_conv2(preact, adjacency,
                                     num_outputs_per_channel=num_outputs_per_channel,
                                     n_channels=n_channels, one_hop=False,
                                     activation_fn=None, scope='shortcut')

        residual = ar_graph_conv2(preact, adjacency,
                                     num_outputs_per_channel=depth_bottleneck,
                                     n_channels=n_channels, one_hop=False, # activation_fn=None,
                                     scope='res/conv1')
        #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact1')

        residual = ar_graph_conv2(residual, adjacency,
                                     num_outputs_per_channel=depth_bottleneck,
                                     n_channels=n_channels, #activation_fn=None,
                                     scope='res/conv2')
        #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact2')
        residual = ar_graph_conv2(residual, adjacency,
                                     num_outputs_per_channel=num_outputs_per_channel,
                                     n_channels=n_channels, one_hop=False,
                                     activation_fn=None,
                                     scope='res/conv3')

        output = activation_fn( shortcut + residual )

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


@add_arg_scope
def graph_bottleneck2(inputs,
                         adjacency,
                         num_outputs,
                         depth_bottleneck,
                         activation_fn=tf.nn.elu,
                         outputs_collections=None, scope=None):
    """
    Bottleneck residual unit variant
    """
    ss, n_features = inputs.get_shape()

    with tf.variable_scope(scope, 'graph_bottleneck2', [inputs, adjacency]) as sc:

        #preact = layers.batch_norm(inputs, activation_fn=activation_fn, scope='preact')
        preact = inputs
        if n_features == num_outputs:
            shortcut = inputs
        else:
            shortcut = graph_conv2(preact, adjacency,
                                     num_outputs=num_outputs, one_hop=False,
                                     activation_fn=None, scope='shortcut')

        residual = graph_conv2(preact, adjacency,
                                     num_outputs=depth_bottleneck, one_hop=False, # activation_fn=None,
                                     scope='res/conv1')
        #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact1')

        residual = graph_conv2(residual, adjacency,
                                     num_outputs=depth_bottleneck, #activation_fn=None,
                                     scope='res/conv2')
        #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact2')
        residual = graph_conv2(residual, adjacency,
                                     num_outputs=num_outputs, one_hop=False,
                                     activation_fn=None,
                                     scope='res/conv3')

        output = activation_fn( shortcut + residual )

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


@add_arg_scope
def graph_bottleneck(inputs,
                     adjacency,
                     depth_in,
                     depth,
                     depth_bottleneck,
                     activation_fn=tf.nn.elu,
                     outputs_collections=None, scope=None):
    """
    Bottleneck residual unit variant
    """
    with tf.variable_scope(scope, 'graph_bottleneck', [inputs, adjacency]) as sc:

        if depth == depth_in:
            shortcut = inputs
        else:
            shortcut = slim.fully_connected(inputs, depth, activation_fn=None, scope='shortcut')

        residual = slim.fully_connected(inputs, depth_bottleneck, scope='res/fc1')
        residual = graph_conv(residual, adjacency, depth_bottleneck, scope='res/conv2')
        residual = slim.fully_connected(residual, depth, activation_fn=None, scope='res/fc2')

        output = activation_fn(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

@add_arg_scope
def ar_graph_bottleneck( inputs,
                         adjacency,
                         num_outputs_per_channel,
                         n_channels,
                         depth_bottleneck,
                         activation_fn=tf.nn.elu,
                         outputs_collections=None, scope=None):
    """
    Bottleneck residual unit variant
    """
    ss, n_features = inputs.get_shape()
    num_inputs_per_channel = tf.cast(n_features / n_channels, tf.int32)

    with tf.variable_scope(scope, 'ar_graph_bottleneck', [inputs, adjacency]) as sc:

        #preact = layers.batch_norm(inputs, activation_fn=activation_fn, scope='preact')
        preact = inputs
        if num_inputs_per_channel == num_outputs_per_channel:
            shortcut = inputs
        else:
            shortcut = ar_graph_conv(preact, adjacency,
                                     num_outputs_per_channel=num_outputs_per_channel,
                                     n_channels=n_channels, one_hop=False,
                                     activation_fn=None, scope='shortcut')

        residual = ar_graph_conv(preact, adjacency,
                                     num_outputs_per_channel=depth_bottleneck,
                                     n_channels=n_channels, one_hop=False, # activation_fn=None,
                                     scope='res/conv1')
       #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact1')

        residual = ar_graph_conv(residual, adjacency,
                                     num_outputs_per_channel=depth_bottleneck,
                                     n_channels=n_channels, #activation_fn=None,
                                     scope='res/conv2')
        #residual =  layers.batch_norm(residual, activation_fn=activation_fn, scope='res/preact2')
        residual = ar_graph_conv(residual, adjacency,
                                     num_outputs_per_channel=num_outputs_per_channel,
                                     n_channels=n_channels, one_hop=False,
                                     activation_fn=None,
                                     scope='res/conv3')

        output = activation_fn( shortcut + residual )

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)

@add_arg_scope
def ar_graph_conv( features,
                adjacency,
                num_outputs_per_channel,
                n_channels,
                activation_fn=tf.nn.elu,
                zero_hop=True,
                one_hop=True,
                selfmask=False,
                ar_features=False,
                weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                dropout=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    """
    Adds a fully connected layer.
    """
    ss, n_features = features.get_shape()
    num_outputs = num_outputs_per_channel * n_channels
    num_inputs_per_channel = tf.cast(n_features / n_channels, tf.int32)

    with tf.variable_scope(scope, 'graph_conv', [features, adjacency], reuse=reuse) as sc:

        if zero_hop:
            w0 = model_variable('weights_0',
                                    shape=[n_features, num_outputs],
                                    initializer=weights_initializer,
                                    regularizer=weights_regularizer,
                                    trainable=True)
        if one_hop:
            w1 = model_variable('weights_1',
                                shape=[n_features, num_outputs],
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=True)

        b = model_variable('bias',
                            shape=[num_outputs],
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=True)

        # Build AR mask for features
        if selfmask or ar_features:
            mask0 = tf.sequence_mask([ ( i / num_inputs_per_channel + 1 ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
        else:
            mask0 = tf.sequence_mask([ ( i / num_inputs_per_channel ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
        mask0 = tf.to_float( tf.logical_not(mask0))

        if ar_features:
            mask1 = tf.sequence_mask([ ( i / num_inputs_per_channel + 1 ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
        else:
            mask1 = tf.sequence_mask([ ( i / num_inputs_per_channel ) * num_outputs_per_channel  for i in range(n_features) ], num_outputs)
        mask1 = tf.to_float( tf.logical_not(mask1))

        if dropout is not None:
            features = tf.nn.dropout(features, keep_prob=dropout)

        if one_hop:
            outputs = tf.matmul(features, tf.multiply(mask1, w1))
            outputs = tf.sparse_tensor_dense_matmul(adjacency, outputs)

            if zero_hop:
                outputs = 0.5*outputs + 0.5*tf.matmul(features, tf.multiply(mask0, w0))
        else:
            outputs = tf.matmul(features, tf.multiply(mask0, w0))

        outputs += b

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def graph_conv( features,
                adjacency,
                num_outputs,
                activation_fn=tf.nn.elu,
                masked=False,
                weights_initializer=slim.variance_scaling_initializer(factor=sqrt(2)),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                dropout=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
    """
    Adds a fully connected layer.
    """
    ss, n_features = features.get_shape()

    with tf.variable_scope(scope, 'graph_conv', [features, adjacency], reuse=reuse) as sc:

        if not masked:
            w0 = model_variable('weights_0',
                                shape=[n_features, num_outputs],
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=True)

        w1 = model_variable('weights_1',
                            shape=[n_features, num_outputs],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            trainable=True)

        b = model_variable('bias',
                            shape=[num_outputs],
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=True)

        if dropout is not None:
            features = tf.nn.dropout(features, keep_prob=dropout)

        outputs = tf.matmul(features, w1)
        outputs = tf.sparse_tensor_dense_matmul(adjacency, outputs)

        if not masked:
            outputs = 0.5*outputs + 0.5*tf.matmul(features, w0)

        outputs += b

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                outputs)

@add_arg_scope
def graph_downsample(features,
                     indices,
                     scope=None):
    """
    Simple downsampling layer
    """
    return tf.gather(features, indices, name='downsample')


@add_arg_scope
def graph_upsample(features,
                   original_indices,
                   original_shape,
                   scope=None):
    """
    Simple upsampling layer with zero insertion
    """
    return tf.scatter_nd(tf.reshape(original_indices, [-1,1]),
                         features,
                         original_shape)
