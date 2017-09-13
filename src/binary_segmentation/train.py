
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import math
import re
import os
import settings
# from input import distorted_inputs
from input import distorted_inputs_files
from input import eval_inputs_files
import unet


import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

modeldir = FLAGS.train_dir + "/" + FLAGS.experiment + "/" + str(FLAGS.crossval_run) + "/"


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def train_model(total_loss, global_step):
  """Train model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = FLAGS.num_examples_per_epoch_train / FLAGS.train_batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  FLAGS.learning_rate_decay_factor,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # # Add histograms for trainable variables.
  # for var in tf.trainable_variables():
  #   tf.histogram_summary(var.op.name, var)
  #
  # # Add histograms for gradients.
  # for grad, var in grads:
  #   if grad is not None:
  #     tf.tf.summary.scalar(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  # variable_averages = tf.train.ExponentialMovingAverage(
  #     FLAGS.moving_average_decay, global_step)
  # variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op




def tower_loss(scope):

    # Get images and labels for CIFAR-10.
    images, labels = distorted_inputs_files(filename=FLAGS.training_file,  batch_size=FLAGS.train_batch_size)


    # images, labels = cifar10.distorted_inputs()

    # Build inference Graph.
    logits = unet.model(x=images, channels=FLAGS.num_channels, n_class=FLAGS.num_classes, layers=FLAGS.unet_layers, features_root=FLAGS.unet_features_root, keep_prob=0.5)
    print(logits.shape)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.

    weights = np.ones([FLAGS.train_batch_size, int(FLAGS.image_height * FLAGS.resize_factor), int(FLAGS.image_width * FLAGS.resize_factor), FLAGS.num_classes])

    if(FLAGS.weighted_loss):
        weights[:,:,:,0] = 1
        # weights[:,:,:,1] = FLAGS.class_weights
        # weights[:,:,:,2] = FLAGS.class_weights
        # weights[:,:,:,3] = FLAGS.class_weights
    else:
        weights[:,:,:,0] = 1
        # weights[:,:,:,1] = 1
        # weights[:,:,:,2] = 1
        # weights[:,:,:,3] = 1

    _ = unet.binary_loss(logits, labels, weights.astype(np.float32))

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')


    # dice = unet.dice(tf.nn.softmax(logits), labels)



    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % FLAGS.TOWER_NAME, '', l.op.name)
        print(loss_name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    print(tower_grads)
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, reset_op



def train():
    """Train unet for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        display_step = 1
        tb_step = 50
        save_step = math.ceil(FLAGS.num_examples_per_epoch_train / FLAGS.train_batch_size)

        print("seed="+str(FLAGS.seed))

        images_train, labels_train = distorted_inputs_files(filename=FLAGS.training_file,  batch_size=FLAGS.train_batch_size)

        images_eval, labels_eval, num_records_eval  = eval_inputs_files(filename=FLAGS.eval_file,  batch_size=FLAGS.train_batch_size)



        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.train_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_channels))

        labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.train_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

        bn_training_placeholder = tf.placeholder(tf.bool)
        keep_prob_training_placeholder = tf.placeholder(tf.float32)


        # Build a Graph that computes the logits predictions from the
        # inference model.
        # Build inference Graph.

        logits = unet.model(x=images_placeholder, channels=FLAGS.num_channels, n_class=FLAGS.num_classes, layers=FLAGS.unet_layers, features_root=FLAGS.unet_features_root, keep_prob=keep_prob_training_placeholder, filter_size=FLAGS.unet_kernel, training=bn_training_placeholder)

        weights = np.ones([FLAGS.train_batch_size, int(FLAGS.image_height * FLAGS.resize_factor), int(FLAGS.image_width * FLAGS.resize_factor), FLAGS.num_classes])
        #
        # weights[:,:,:,0] = 1
        # weights[:,:,:,1] = 2


        if(FLAGS.loss_type == "dice"):
            loss_ = 1 - unet.dice_coe(tf.nn.softmax(logits), labels_placeholder)
        else:
            loss_ = unet.weighted_loss(logits, labels_placeholder, weights.astype(np.float32))
        # loss_ = 1 - unet.dice_coe(tf.nn.softmax(logits), labels_placeholder)

        predictions = tf.argmax(tf.nn.softmax(logits), axis=3)
        # mean_iou = tf.metrics.mean_iou(tf.argmax(labels_placeholder, axis=3),predictions, FLAGS.num_classes)

        # logits_sigmoid = tf.nn.sigmoid(logits)
        # logits_inv = 1 - logits_sigmoid
        #
        # logits_for_iou = tf.concat([logits_sigmoid, logits_inv],axis=3)
        # print(logits_for_iou)


        mean_iou, epoch_mean_iou_reset = create_reset_metric(
                    tf.metrics.mean_iou, 'mean_iou_loss',
                    predictions=predictions, labels=tf.argmax(labels_placeholder, axis=3),num_classes=FLAGS.num_classes )



        train_op = train_model(loss_, global_step)

        saver = tf.train.Saver( max_to_keep=5)  #keep all checkpoints
        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge_all()

        best_eval = 0

        with tf.Session() as sess:
            print("Starting Training Loop")
            sess.run(tf.local_variables_initializer())
            print("Local Variable Initializer done...")
            sess.run(tf.global_variables_initializer())
            print("Global Variable Initializer done...")
            coord = tf.train.Coordinator()
            print("Train Coordinator done...")
            print("Starting Queue Runner")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(modeldir, sess.graph)
            # train_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)
            # evaluation_tot = [0,0,0,0]
            for step in range(FLAGS.max_steps):
                    start_time = time.time()
                    images_s, labels_s= sess.run([images_train, labels_train])


                    assert not np.isnan(images_s).any(), 'Images diverged with loss = NaN'
                    assert not np.isnan(labels_s).any(), 'Labels diverged with loss = NaN'
                    train_feed = {images_placeholder: images_s,
                                    labels_placeholder: labels_s,
                                    bn_training_placeholder: True,
                                    keep_prob_training_placeholder: 0.8}


                    if step % 100 == 0:
                        _,loss_value, logits_value, summary_str = sess.run([ train_op, loss_, logits, summary_op], feed_dict=train_feed)
                        summary_writer.add_summary(summary_str, step)
                    else:
                        _,loss_value, logits_value = sess.run([ train_op, loss_, logits], feed_dict=train_feed)

                    assert not np.isnan(logits_value).any(), 'Logits diverged with loss = NaN'
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    duration = time.time() - start_time

                    num_examples_per_step = FLAGS.train_batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                                        'sec/batch)')
                    print_str_loss = format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch)
                    print (print_str_loss)

                    # Save the model checkpoint periodically.
                    if step % save_step == 0 or (step + 1) == FLAGS.max_steps:


                        eval_loss = []
                        start_time = time.time()
                        sess.run(epoch_mean_iou_reset)
                        for eval_step in range(int(num_records_eval/FLAGS.train_batch_size)):

                            images_s, labels_s= sess.run([images_eval, labels_eval])
                            eval_feed = {images_placeholder: images_s,
                                              labels_placeholder: labels_s,
                                              bn_training_placeholder: False,
                                              keep_prob_training_placeholder: 1.0}

                            loss_value, logits_value, mean_iou_value = sess.run([ loss_, logits, mean_iou], feed_dict=eval_feed)

                            print(mean_iou_value)
                            eval_loss.append(loss_value)


                        duration = time.time() - start_time
                        num_examples_per_step = int(num_records_eval/FLAGS.train_batch_size)
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        loss_value = np.nanmean(np.array(eval_loss))
                        format_str = ('Eval Step: %s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                                                                'sec/batch)')
                        print_str_loss = format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch)
                        print (print_str_loss)

                        summary = tf.Summary()
                        summary.value.add(tag='xentropy', simple_value=loss_value)
                        summary.value.add(tag='mean_iou', simple_value=mean_iou_value[0])
                        iou_instrument = mean_iou_value[1][1,1] / (mean_iou_value[1][1,1]  + mean_iou_value[1][0,1] + mean_iou_value[1][1,0]  )
                        print(iou_instrument)
                        summary.value.add(tag='instrument_iou', simple_value=iou_instrument)
                        summary_writer.add_summary(summary, step)

                        if(mean_iou_value[0] > best_eval):
                            print("Saving Epoch + Evaluation")
                            checkpoint_path = os.path.join(modeldir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
                            best_eval = mean_iou_value[0]




            # print('Model saved in file: ' + save_path)
            coord.request_stop()
            coord.join(threads)







def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(modeldir):
        tf.gfile.DeleteRecursively(modeldir)
    tf.gfile.MakeDirs(modeldir)
    train()


if __name__ == '__main__':
    tf.app.run()
