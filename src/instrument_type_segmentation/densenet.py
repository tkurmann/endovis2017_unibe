from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
from collections import OrderedDict


FLAGS = tf.app.flags.FLAGS


def weight_variable(name, shape, stddev=0.1, wd=1e-4):
    # with tf.device('/gpu:0'):
    dtype = tf.float32
    # var = tf.get_variable(name, initializer=tf.truncated_normal(shape, stddev=stddev), dtype=dtype)
    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=FLAGS.seed), dtype=dtype)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    return var

def weight_variable_deconv(name, shape, stddev=0.1, wd=1e-4):
    # with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=FLAGS.seed), dtype=dtype)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    return var

def bias_variable(name, shape):
    # with tf.device('/gpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=FLAGS.seed), dtype=dtype)
    return var


def conv2d(x, W,keep_prob_=0.5, pad=True):

    # if pad:
    #     x = tf.pad(x, [[0,0],[1, 1,], [1, 1],[0,0]], "SYMMETRIC")
    return  tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #return tf.nn.dropout(conv_2d, keep_prob_)

def dilated_conv2d(x, W, dilation_rate):

    return  tf.nn.convolution(x, W, dilation_rate=[dilation_rate,dilation_rate], padding='SAME')



def deconv2d(x, W,stride, pad=False):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*stride, x_shape[2]*stride, x_shape[3]//2])
    out =  tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
    # if pad:
    #     out = tf.pad(x, [[0,0],[1, 1,], [1, 1],[0,0]], "SYMMETRIC")
    return out

def max_pool(x,n, pad=False):
    if pad:
            x = tf.pad(x, [[0,0],[1, 1,], [1, 1],[0,0]], "SYMMETRIC")
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


def _variable_on_cpu(name, shape, initializer, trainable=True):
    # with tf.device('/gpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)

    return var


def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed





def res_block(num_channels, prev_num_channels, input, filter_size, train=True):

    stddev = tf.sqrt(2 / (filter_size**2 * num_channels))

    b_1 = bias_variable("b1",[num_channels])
    w_1 = weight_variable("w1", [filter_size, filter_size, prev_num_channels, num_channels], stddev)

    conv_1 = dilated_conv2d(input, w_1, dilation_rate=2)
    bn_1 = tf.contrib.layers.batch_norm(inputs = conv_1+b_1, decay=0.9, is_training=train, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None, fused=True)

    b_2 = bias_variable("b2",[num_channels])
    w_2 = weight_variable("w2", [filter_size, filter_size, num_channels, num_channels], stddev)


    conv_2 = dilated_conv2d(bn_1, w_2, dilation_rate=2)
    bn_2 = tf.contrib.layers.batch_norm(inputs = conv_2+b_2, decay=0.9, is_training=train, center=True, scale=True, activation_fn=None, updates_collections=None, fused=True)



    if(prev_num_channels != num_channels):
        b_s = bias_variable("bs",[num_channels])
        w_s = weight_variable("ws", [filter_size, filter_size, prev_num_channels, num_channels], stddev)
        shortcut = conv2d(input,w_s)
        shortcut = tf.contrib.layers.batch_norm(inputs=shortcut+b_s, decay=0.9, is_training=train, center=True, scale=True, activation_fn=None, updates_collections=None, fused=True)
    else:
        shortcut = input

    output = tf.nn.relu(shortcut + bn_2)
    return output




def layer(input, num_channels,prev_num_channels,  filter_size, dropout_ratio, training):

    bn = tf.contrib.layers.batch_norm(inputs = input, decay=0.9, is_training=training, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None, fused=True)

    stddev = tf.sqrt(2 / (filter_size**2 * num_channels))
    b_1 = bias_variable("b1",[num_channels])
    w_1 = weight_variable("w1", [filter_size, filter_size, prev_num_channels, num_channels], stddev)

    conv = tf.nn.conv2d(bn, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1

    return tf.nn.dropout(conv, dropout_ratio)


def dense_block(input, layers, input_num_channels, growth_rate, dropout_ratio, training, filter_size):

    layer_outputs = OrderedDict()
    output = []
    c = input
    num_channels = growth_rate
    prev_num_channels = input_num_channels

    for layer_num in range(layers):
        with tf.variable_scope('layer_'+str(layer_num)) as scope:
            print(layer_num)
            layer_outputs[layer_num] = layer(c, num_channels=num_channels, prev_num_channels=prev_num_channels , filter_size=filter_size, dropout_ratio=dropout_ratio, training=training)

        c = tf.concat((c,layer_outputs[layer_num]), axis=3)
        prev_num_channels += growth_rate


    for layer_num in range(layers):
        if(layer_num == 0):
            output = layer_outputs[layer_num]
        else:
            output = tf.concat((output,layer_outputs[layer_num]), axis=3)

    return output

def transition_down(input, num_channels, dropout_ratio, training):

    filter_size = 1
    n_max_pool = 2
    print(num_channels)
    bn = tf.contrib.layers.batch_norm(inputs = input, decay=0.9, is_training=training, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None, fused=True)

    stddev = tf.sqrt(2 / (filter_size**2 * num_channels))
    b_1 = bias_variable("b1",[num_channels])
    w_1 = weight_variable("w1", [filter_size, filter_size, num_channels, num_channels], stddev)

    conv = tf.nn.conv2d(bn, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1

    do = tf.nn.dropout(conv, dropout_ratio)

    return tf.nn.max_pool(do, ksize=[1, n_max_pool, n_max_pool, 1], strides=[1, n_max_pool, n_max_pool, 1], padding='SAME')


def transition_up(input, num_channels,output_shape):

    filter_size = 3
    stride = 2
    stddev = tf.sqrt(2 / (filter_size**2 * num_channels))
    b_1 = bias_variable("b1",[num_channels])
    w_1 = weight_variable("w1", [filter_size, filter_size, num_channels, num_channels], stddev)

    return  tf.reshape(tf.nn.conv2d_transpose(input,
                            w_1,
                            output_shape,
                            [1,stride,stride,1],
                            padding='SAME'), shape=output_shape)




def model(x, channels, n_class, layers=3, features_root=64, filter_size=3, pool_size=2, summaries=True, two_sublayers=True, training=True, keep_prob=1.0):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    # training = tf.constant(training)
    print("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    bs = tf.shape(x)[0]
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    #x_image = tf.reshape(x, tf.pack([-1,nx,ny,channels]))
    in_node = x
    #batch_size = tf.shape(x_image)[0]

    down_layers = OrderedDict()
    up_layers = OrderedDict()
    n_layers_per_dense = 4
    growth_rate = 12



    with tf.variable_scope('input') as scope:
        features_root = 48
        stddev = tf.sqrt(2 / (1**2 * features_root))
        weight = weight_variable("w",[1, 1, channels, features_root], stddev)
        bias = bias_variable("b", [features_root])
        conv = conv2d(in_node, weight, tf.constant(1.0), pad=False)
        input = conv + bias



    with tf.variable_scope('encoder') as scope:
        for nd_layer in range(layers):
            with tf.variable_scope('dense_layer_'+str(nd_layer)) as scope:

                num_channels_input = features_root + nd_layer*growth_rate*n_layers_per_dense
                print(num_channels_input)
                dblock = dense_block(input, n_layers_per_dense, num_channels_input, growth_rate, keep_prob, training, filter_size=3)

                down_layers[nd_layer] = tf.concat((dblock, input), axis=3)
                print(down_layers[nd_layer])

                input = transition_down(down_layers[nd_layer], features_root + (nd_layer+1)*growth_rate*n_layers_per_dense, keep_prob, training)
                print(input)

    with tf.variable_scope('bottleneck') as scope:

            num_channels_input = features_root + layers*growth_rate*n_layers_per_dense
            print(num_channels_input)

            dblock = dense_block(input, n_layers_per_dense, num_channels_input, growth_rate, keep_prob, training, filter_size=3)

            # input = tf.concat((dblock, input), axis=3)
            input = dblock


    with tf.variable_scope('decoder') as scope:
        for nd_layer in range(layers):
            with tf.variable_scope('dense_layer_'+str(nd_layer)) as scope:

                num_channels_input = features_root + (layers-nd_layer+1)*growth_rate*n_layers_per_dense
                print(num_channels_input)
                print(input)

                tup_channels = growth_rate*n_layers_per_dense

                input = transition_up(input, tup_channels, output_shape=([bs,tf.cast(nx/(2**(layers-nd_layer-1)),tf.int32),tf.cast(ny/(2**(layers-nd_layer-1)),tf.int32), tup_channels]))

                print(input)
                print(down_layers[(layers-nd_layer-1)])

                up_layers[(layers-nd_layer)] = tf.concat((input, down_layers[(layers-nd_layer-1)]), axis=3)
                print(up_layers[(layers-nd_layer)])

                input = dense_block(up_layers[(layers-nd_layer)], n_layers_per_dense, num_channels_input, growth_rate, keep_prob, training, filter_size=3)





    # Output Map
    with tf.variable_scope('output') as scope:
        stddev = tf.sqrt(2 / (1**2 * features_root))
        weight = weight_variable("w",[1, 1, 48, n_class], stddev)
        bias = bias_variable("b", [n_class])
        conv = conv2d(input, weight, tf.constant(1.0), pad=False)
        output_map = conv + bias
        # up_h_convs["out"] = output_map


    return output_map



def loss(logits, labels, weights):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  labels = tf.squeeze(labels)
  print(labels.shape)

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  # / ( FLAGS.train_batch_size * 256 *256)
  # cross_entropy_mean = tf.constant([-1.0])
  tf.add_to_collection('losses', cross_entropy_mean)
  # tf.add_to_collection('losses_2', cross_entropy_mean_2)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).

  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def binary_loss(logits, labels, weights):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  # labels = tf.squeeze(labels)
  # print(labels.shape)
  #
  # weights = tf.constant(weights)
  #
  # logits = tf.nn.sigmoid(logits)
  #
  # logits = tf.maximum(logits, 10e-10)
  # logits = tf.minimum(logits, 1-10e-10)
  #
  #
  #
  # cross_entropy = labels * tf.log(logits) + ((1-labels) * tf.log(1-logits))
  #
  # cross_entropy = tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy_per_example') / (FLAGS.image_width*FLAGS.resize_factor)
  #
  # cross_entropy = tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy_per_example')/ (FLAGS.image_height*FLAGS.resize_factor)
  #
  # # x = tf.Print(cross_entropy, [cross_entropy], summarize=100)
  #
  # cross_entropy_mean = -tf.reduce_mean(cross_entropy, name='cross_entropy')
  # / ( FLAGS.train_batch_size * 256 *256)
  # cross_entropy_mean = tf.constant([-1.0])
  cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
  print(cross_entropy_mean.shape)
  tf.add_to_collection('losses', cross_entropy_mean)
  # tf.add_to_collection('losses_2', cross_entropy_mean_2)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).

  return tf.add_n(tf.get_collection('losses'), name='total_loss')




def dice_coe(output, target, loss_type='jaccard', axis=[1,2,3], smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both output and target are empty, it makes sure dice is 1.
        If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
        so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice



def mse_loss(logits, labels, weights):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  # labels = tf.squeeze(labels)
  print(labels.shape)

  weights = tf.constant(weights)

  logits = tf.nn.sigmoid(logits)

  logits = tf.maximum(logits, 10e-10)
  logits = tf.minimum(logits, 1-10e-10)


  l2 = tf.sqrt(tf.pow((labels - logits),2))
  l2 = tf.reduce_sum(l2, axis=1, name='cross_entropy_per_example') / (FLAGS.image_width*FLAGS.resize_factor)

  l2 = tf.reduce_sum(l2, axis=1, name='cross_entropy_per_example')/ (FLAGS.image_height*FLAGS.resize_factor)
  l2_mean = tf.reduce_mean(l2, name='cross_entropy')
  # / ( FLAGS.train_batch_size * 256 *256)
  # cross_entropy_mean = tf.constant([-1.0])
  tf.add_to_collection('losses', l2_mean)
  # tf.add_to_collection('losses_2', cross_entropy_mean_2)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).

  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def rates(logits, labels, channel=0):

        predictions = tf.argmax(logits, axis=3)
        predictions = tf.cast(tf.equal(predictions,channel), tf.float32)


        actuals = labels[:,:,:,channel]

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
        )

        tn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
        )

        fp_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ),
          "float"
        )
        )

        fn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ),
          "float"
        )
        )

        return [tp_op,tn_op,fp_op,fn_op]


def dice_loss(logits, labels, channel=0):

    r = rates(logits, labels, channel)
    dice = 2 * r[0] / (2* r[0] + r[2] + r[3])
    return dice,
