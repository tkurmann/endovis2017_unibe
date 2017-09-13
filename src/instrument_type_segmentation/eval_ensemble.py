import tensorflow as tf
from scipy import misc
import numpy as np
import glob
import sys
import os
import unet
import settings
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax,unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from input import eval_inputs_files

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


tf.app.flags.DEFINE_string('image_folder', '../../data_export/data/instrument_dataset_1/left_frames',
                                """Folder with data for inference""")

tf.app.flags.DEFINE_string('models_folder', 'model/train/parts_ensemble_0',
                                """Folder with models""")

FLAGS = tf.app.flags.FLAGS


def crop_frame(image):

    im_shape = image.shape

    min_x = 320
    min_y = 0
    max_x = min_x+FLAGS.image_width
    max_y = min_y+1080 # limited due to height...
    if(len(image.shape) == 3):
        image_new = np.zeros((FLAGS.image_height, FLAGS.image_width, im_shape[2]))
        image_new[100:100+1080, :, :] = image[:,min_x:max_x,:]
    else:
        image_new = np.zeros((FLAGS.image_height, FLAGS.image_width))
        image_new[100:100+1080, :] = image[:,min_x:max_x]

    return image_new

def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, reset_op


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

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
    n_classes = 2
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




def inference():

    models_list = glob.glob(FLAGS.models_folder+"/*")

    images, labels, num_records  = eval_inputs_files(filename=FLAGS.eval_file,  batch_size=FLAGS.eval_batch_size)

    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_channels))

    labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    logits_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    logits = unet.model(x=images_placeholder, channels=FLAGS.num_channels, n_class=FLAGS.num_classes, layers=FLAGS.unet_layers, features_root=FLAGS.unet_features_root, training=False,filter_size=FLAGS.unet_kernel)


    logits = tf.nn.softmax(logits)


    predictions = tf.py_func(dense_crf, [logits_placeholder, tf.cast(images_placeholder*255,tf.uint8)], tf.float32)

    predictions = tf.argmax(logits_placeholder, dimension=3)


    pred = tf.reshape(predictions, [-1,])
    gt = tf.reshape(tf.argmax(labels_placeholder,axis=3), [-1,])



    mean_iou, epoch_mean_iou_reset = create_reset_metric(
            tf.metrics.mean_iou, 'mean_iou_loss',
            predictions=pred, labels=gt,num_classes=FLAGS.num_classes )



    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()


    logits_total = np.zeros((num_records,int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    labels_total = np.zeros((num_records,int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    images_total = np.zeros((num_records,int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_channels))

    for m in range(len(models_list)):
        model = models_list[m]
        print(model)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model)
            if ckpt and ckpt.model_checkpoint_path:
                print("Loading model form Checkpoint")
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("Step:" + global_step)
            else:
                print('No checkpoint file found')
                global_step = -1
                return global_step

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Starting queue runners")
            coord = tf.train.Coordinator()
            try:
                coord = tf.train.Coordinator()
                print("Train Coordinator done...")
                print("Starting Queue Runner")
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            for i in range(num_records):

                image, label = sess.run([images, labels])
                labels_total[i,:,:,:] = label
                images_total[i,:,:,:] = image
                for r in range(5):
                    if(r == 0):
                        images_feed = image
                    elif(r == 1):
                        images_feed = np.fliplr(image)
                    elif(r == 2):
                        images_feed = np.flipud(image)
                    elif(r == 3):
                        images_feed = np.fliplr(np.flipud(image))
                    elif(r == 4):
                        images_feed = np.rot90(image, axes=(1,2))


                    feed = {images_placeholder: images_feed}
                    logits_value_r = sess.run(logits,feed_dict=feed)


                    if(r == 0):
                        logits_value_r = logits_value_r
                    elif(r == 1):
                        logits_value_r = np.fliplr(logits_value_r)
                    elif(r == 2):
                        logits_value_r = np.flipud(logits_value_r)
                    elif(r == 3):
                        logits_value_r = np.fliplr(np.flipud(logits_value_r))
                    elif(r == 4):
                        logits_value_r = np.rot90(logits_value_r,3,axes=(1,2))

                    logits_total[i,:,:,:] += logits_value_r[0,:,:,:]
                    # print(logits_total[i,:,:,:])




            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        try:
            coord = tf.train.Coordinator()
            print("Train Coordinator done...")
            print("Starting Queue Runner")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.local_variables_initializer())

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        for i in range(num_records):
            logits_value = logits_total[i,:,:,:] / (5*len(models_list))  # average over models and rotations
            loss_feed = {logits_placeholder: np.expand_dims(logits_value,0),
                        labels_placeholder: np.expand_dims(labels_total[i,:,:,:],0),
                        images_placeholder: np.expand_dims(images_total[i,:,:,:],0)}

            predictions_value, miou_value = sess.run([predictions, mean_iou],feed_dict=loss_feed)
            print(miou_value)

            # fig = plt.figure(figsize=(20, 10))
            # plt.axis('off')
            # plt.imshow(predictions_value[0,:,:], vmin=0, vmax=1.0)
            # plt.savefig('inference/image' + str(i) + '.png',bbox_inches='tight')
            # plt.close()

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):  # pylint: disable=unused-argument
  inference()


if __name__ == '__main__':
    tf.app.run()
