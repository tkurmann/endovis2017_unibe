import tensorflow as tf
from scipy import misc
from scipy import ndimage
from joblib import Parallel, delayed
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




ORIGINAL_HEIGHT = 1080
ORIGINAL_WIDTH = 1920


tf.app.flags.DEFINE_string('image_folder', '../../data_export/data/instrument_dataset_1/left_frames',
                                """Folder with data for inference""")

tf.app.flags.DEFINE_string('inference_file', '../../data_export/tf2/binary_train_0.tfrecords',
                            """CSV with training data""")

tf.app.flags.DEFINE_string('output_folder', 'results/',
                            """CSV with training data""")

tf.app.flags.DEFINE_string('models_folder', 'model/train/no_probe',
                                """Folder with models""")
tf.app.flags.DEFINE_integer('frame_start_number', 225,
                                """name for output""")

FLAGS = tf.app.flags.FLAGS

# logits_global = []

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





def save_results(i, logits_tot):

    # global logits_global
    # print(i)

    output_logits_background = np.ones((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 1))
    output_logits_instrument = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, FLAGS.num_classes-1))
    output_logits = np.concatenate((output_logits_background, output_logits_instrument), axis=2)
    logits = logits_tot[i,:,:,:]
    print(np.sum(logits))
    print(logits.shape)
    start_y = int(100 * FLAGS.resize_factor)
    stop_y = int(start_y + 1080*FLAGS.resize_factor)
    print(start_y)
    print(stop_y)
    logits_no_borders = logits[start_y:stop_y,:,:]

    print(np.min(logits_no_borders))
    print(np.max(logits_no_borders))

    logits_resized = ndimage.interpolation.zoom(logits_no_borders, zoom=(1/FLAGS.resize_factor,1/FLAGS.resize_factor,1), mode='nearest')

    print(np.min(logits_resized))
    print(np.max(logits_resized))

    output_logits[:,320:320+FLAGS.image_width,:] = logits_resized


    predictions = np.argmax(output_logits, axis=2)
    predictions = predictions.astype(np.uint8)

    misc.imsave(FLAGS.output_folder +"frame" + str(FLAGS.frame_start_number+i)+".png", predictions)

    misc.imsave(FLAGS.output_folder +"frame_" + "logits_"+str(FLAGS.frame_start_number+i)+".png", np.clip(output_logits[:,:,0],0.0, 1.0))


    return 0




def inference():

    models_list = glob.glob(FLAGS.models_folder+"/*")

    images, labels, num_records  = eval_inputs_files(filename=FLAGS.inference_file,  batch_size=FLAGS.eval_batch_size)

    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_channels))

    logits_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.eval_batch_size, int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

    logits = unet.model(x=images_placeholder, channels=FLAGS.num_channels, n_class=FLAGS.num_classes, layers=FLAGS.unet_layers, features_root=FLAGS.unet_features_root, training=False,filter_size=FLAGS.unet_kernel)

    logits = tf.nn.softmax(logits)

    predictions = tf.py_func(dense_crf, [logits_placeholder, tf.cast(images_placeholder*255,tf.uint8)], tf.float32)

    predictions = tf.argmax(logits_placeholder, dimension=3)


    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver()

    logits_total = np.zeros((num_records,int(FLAGS.image_height*FLAGS.resize_factor), int(FLAGS.image_width*FLAGS.resize_factor), FLAGS.num_classes))

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
                # labels_total[i,:,:,:] = label
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

                    logits_total[i,:,:,:] += logits_value_r[0,:,:,:]/(5)
                    # print(logits_total[i,:,:,:])




            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)



            # logits_global = logits_total
            # print(len(logits_global))

    nsamples = num_records
    start = list(range(0,nsamples, round(nsamples/8)))
    stop = list(range(round(nsamples/8),nsamples+(nsamples%round(nsamples/8)), round(nsamples/8)))
    args = np.concatenate((start,stop))

    arg_instances = list(range(nsamples))
    results = Parallel(n_jobs=-1, backend="threading")(delayed(save_results)(i, logits_total) for i in range(nsamples))





def main(unused_argv):
    print(FLAGS.inference_file)
    tf.gfile.MakeDirs(FLAGS.output_folder )
    inference()


if __name__ == '__main__':
    tf.app.run(main=main)
