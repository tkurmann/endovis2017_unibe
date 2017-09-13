from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import itertools

import settings


from input import eval_inputs_files
from input import distorted_inputs_files

import unet
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax,unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once',
                            False,
                            """Whether to run eval only once.""")



modeldir_train = FLAGS.train_dir + "/" + FLAGS.experiment + "/" + str(FLAGS.crossval_run) + "/"

modeldir_eval = FLAGS.eval_dir + "/" + FLAGS.experiment + "/" + str(FLAGS.crossval_run) + "/"



def to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret



def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=1,
              srgb_bilateral=(15, 15, 15),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape
    n_classes = FLAGS.num_classes
    probs = probs[0].transpose(2, 0, 1) # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = unary_from_softmax(probs)
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
    # return probs



# def my_func(x):
#   # x will be a numpy array with the contents of the placeholder below
#   return np.sinh(x)




def plot_results(images_value, labels_value, logits_value, crf_value,  step):


        # predictions = logits_value
        # predictions = np.argmax(logits_value, axis=3)
        labels = labels_value
        # predictions = np.squeeze(predictions)
        labels = np.squeeze(labels)
        print(logits_value.shape)
        # idx_tp = np.equal(predictions, labels)

        # images_value = to_rgb(images_value[0,:,:,0])

        fig = plt.figure(figsize=(20, 10))
        plt.subplot(141)
        plt.imshow(images_value[0,:,:,0:3], vmin=0, vmax=1.0)
        plt.axis('off')

        # idx_dis = logits_value[0,:,:,0] < 0.5
        # mask = np.zeros(images_value[0,:,:,0].shape)

        # print(labels.shape)
        # print(predictions.shape)

        # for i in range(0,idx_tp.shape[0]):
        #     for j in range(0,idx_tp.shape[1]):
        #         # print(predictions[i,j] )
        #         # print(labels[i,j])
        #         if(predictions[i,j] == labels[i,j] and labels[i,j] > 0): #TP
        #             images_value[i,j] = [0,1.0,0]
        #         elif(~idx_tp[i,j] and labels[i,j] > 0 and predictions[i,j] > 0): #WRONG CLASS
        #             images_value[i,j] = [1.0,0,0]
        #         elif(~idx_tp[i,j] and labels[i,j] == 0 and predictions[i,j] > 0): #OVERSEGMENT
        #             images_value[i,j] = [0,0,1.0]
        #         elif(~idx_tp[i,j] and predictions[i,j] == 0 and labels[i,j] > 0): #OVERSEGMENT
        #             images_value[i,j] = [0,1.0,1.0]


        plt.subplot(142)
        plt.imshow(images_value[0,:,:,3], vmin=0, vmax=1.0)
        plt.axis('off')
        # plt.imshow(mask, cmap='jet', alpha=0.5)

        plt.subplot(143)
        plt.imshow(crf_value[0,:,:]*30, vmin=0, vmax=255)
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(np.argmax(labels_value[0,:,:,:],axis=2)*30, vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig('test/image' + str(step) + '.png',bbox_inches='tight')
        plt.close()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def eval_once(saver, summary_writer, summary_op, num_records, old_step, images, labels, logits, loss, predictions, miou, images_placeholder,logits_placeholder, labels_placeholder):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.eval_gpu_fraction)
    with tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=4, gpu_options=gpu_options)) as sess:

        ckpt = tf.train.get_checkpoint_state(modeldir_train)
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading model form Checkpoint")
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print("Step:" + global_step)
    else:
        print('No checkpoint file found')
        global_step = -1
        return global_step


    if int(global_step) > int(old_step):
        # Start the queue runners.
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        # sess.run(tf.global_variables_initializer())
        #
        print("Starting queue runners")
        coord = tf.train.Coordinator()
        try:
            coord = tf.train.Coordinator()
            print("Train Coordinator done...")
            print("Starting Queue Runner")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(num_records / FLAGS.eval_batch_size))

            total_sample_count = num_iter * FLAGS.eval_batch_size
            step = 0
            losses = []
            dices = []
            while step < num_iter and not coord.should_stop():

                start_time = time.time()
                images_s, labels_value = sess.run([images, labels])
                images_value = images_s
                logits_value = np.zeros((int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

                logits_value_sigmoid = np.zeros((int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

                print("logits_value" + str(logits_value.shape))

                for r in range(4):

                    if(r == 0):
                        images_feed = images_s
                    elif(r == 1):
                        images_feed = np.fliplr(images_s)
                    elif(r == 2):
                        images_feed = np.flipud(images_s)
                    elif(r == 3):
                        images_feed = np.fliplr(np.flipud(images_s))

                    print("images_feed" + str(images_feed.shape))
                    eval_feed = {images_placeholder: images_feed}
                    logits_value_r = sess.run(logits,feed_dict=eval_feed)
                    print("logits_value_r" + str(np.array(logits_value_r).shape))

                    # logits_value_r = sigmoid(logits_value_r)
                    if(r == 0):
                        logits_value_r = logits_value_r
                    elif(r == 1):
                        logits_value_r = np.fliplr(logits_value_r)
                    elif(r == 2):
                        logits_value_r = np.flipud(logits_value_r)
                    elif(r == 3):
                        logits_value_r = np.fliplr(np.flipud(logits_value_r))

                    logits_value = logits_value + logits_value_r
                    logits_value_sigmoid = logits_value_sigmoid + sigmoid(logits_value_r)

                logits_value /= 4
                logits_value_sigmoid /= 4
                print(logits_value.shape)


                loss_feed = {logits_placeholder: logits_value,
                             labels_placeholder: labels_value,
                             images_placeholder: images_feed}

                loss_value, predictions_value, miou_value = sess.run([loss,  predictions, miou],feed_dict=loss_feed)
                print("predictions")
                print(predictions_value.shape)
                # # crf_value = (logits_value > 0.25) + 1
                # #
                # crf_logits = np.concatenate((logits_value_sigmoid[0,:,:,:], 1-logits_value_sigmoid[0,:,:,:]), axis=2)
                # crf_logits = np.transpose(crf_logits, (2,0,1))
                # print(crf_logits.shape)
                # n_labels = 2
                # d = dcrf.DenseCRF2D(640, 512, n_labels)
                #
                # # get unary potentials (neg log probability)
                # U = unary_from_softmax(crf_logits)
                # d.setUnaryEnergy(U)
                #
                # # # This adds the color-independent term, features are the locations only.
                # d.addPairwiseGaussian(sxy=(9, 9), compat=3, kernel=dcrf.DIAG_KERNEL,
                #                            normalization=dcrf.NORMALIZE_SYMMETRIC)
                #
                # print((images_value[0,:,:,:]*255).astype(np.uint8).shape)
                # # # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                # d.addPairwiseBilateral(sxy=(50, 50), srgb=(20, 20, 20), rgbim=(images_value[0,:,:,:]*255).astype(np.uint8),
                #                             compat=10,
                #                             kernel=dcrf.DIAG_KERNEL,
                #                             normalization=dcrf.NORMALIZE_SYMMETRIC)
                #
                # Q = d.inference(5)
                # map = np.argmax(Q, axis=0)
                # crf_value = map.reshape((512,640))
                # # crf_value = logits_value



                plot_results(images_value, labels_value, logits_value_sigmoid, predictions_value, step)
                # dice_value = 2*rates_value[:,0] / (2*rates_value[:,0] +rates_value[:,2] + rates_value[:,3])


                losses.append(loss_value)
                print(miou_value)
                # dices.append(dice_value)


                duration = time.time() - start_time
                step += 1
                num_examples_per_step = FLAGS.eval_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))



        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


        # dice = np.array(dices)
        # print(dice.shape)
        loss = np.array(losses)
        print(np.nanmean(loss, axis=0))




        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        # for ch in range(0,4):
        #     summary.value.add(tag='bscan_dice/'+str(ch), simple_value=np.nanmean(dice[:,ch]))


        summary.value.add(tag='xentropy', simple_value=np.nanmean(loss, axis=0))
        summary_writer.add_summary(summary, global_step)
    else:
        print("Step already computed")
    return global_step


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, reset_op


def evaluate():
  with tf.Graph().as_default() as g:

    images, labels, num_records  = eval_inputs_files(filename=FLAGS.eval_file,  batch_size=FLAGS.eval_batch_size)

    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_channels))

    labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    logits_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    logits = unet.model(x=images_placeholder, channels=FLAGS.num_channels, n_class=FLAGS.num_classes, layers=FLAGS.unet_layers, features_root=FLAGS.unet_features_root, training=False,filter_size=FLAGS.unet_kernel)

    # logits = tf.nn.sigmoid(logits)


    weights = np.ones([FLAGS.train_batch_size, int(FLAGS.image_height * FLAGS.resize_factor), int(FLAGS.image_width * FLAGS.resize_factor), FLAGS.num_classes])
    # weights[:,:,:,0] = 1
    # weights[:,:,:,1] = 1
    # weights[:,:,:,2] = 1
    # weights[:,:,:,3] = 1

    loss = unet.loss(logits_placeholder, labels_placeholder, weights.astype(np.float32))


    logits_sigmoid = tf.nn.sigmoid(logits_placeholder)

    # predictions = tf.round(tf.nn.sigmoid(logits_placeholder))
    # crf_logits = tf.concat((logits_sigmoid, 1-logits_sigmoid), axis=3)
    crf_logits = tf.nn.softmax(logits_placeholder)
    # predictions = tf.argmax(crf_logits, dimension=3)
    predictions = tf.py_func(dense_crf, [crf_logits, tf.cast(images_placeholder[:,:,:,0:3]*255,tf.uint8)], tf.float32)
    predictions = tf.argmax(predictions, dimension=3)
    # # predictions = tf.expand_dims(predictions, dim=3)
    #
    #
    pred = tf.reshape(predictions, [-1,])
    gt = tf.reshape(tf.argmax(labels_placeholder,axis=3), [-1,])
    # print(predictions)


    mean_iou, epoch_mean_iou_reset = create_reset_metric(
                tf.metrics.mean_iou, 'mean_iou_loss',
                predictions=pred, labels=gt,num_classes=FLAGS.num_classes )





    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(modeldir_eval, g)
    old_step = -1
    while int(old_step) < (FLAGS.max_steps-1):
      old_step = eval_once(saver, summary_writer, summary_op, num_records, old_step, images, labels, logits, loss, predictions, mean_iou, images_placeholder, logits_placeholder, labels_placeholder)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(modeldir_eval):
    tf.gfile.DeleteRecursively(modeldir_eval)
  tf.gfile.MakeDirs(modeldir_eval)
  evaluate()


if __name__ == '__main__':
    tf.app.run()
