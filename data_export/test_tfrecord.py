import numpy as np
import scipy.misc
from glob import glob
import pdb
import time
import tensorflow as tf
import argparse
import os
import sys
import math

tf.app.flags.DEFINE_integer('image_height', 1280,
                                """Number of classes.""")
tf.app.flags.DEFINE_integer('image_width', 1280,
                                """Number of classes.""")
tf.app.flags.DEFINE_float('resize_factor', 0.4,
                                    """Number of classes.""")
tf.app.flags.DEFINE_integer('num_channels', 4,
                                """Number of channels in input image.""")
tf.app.flags.DEFINE_integer('num_classes', 7,
                                """Number of channels in input image.""")
tf.app.flags.DEFINE_string('file', 'tf/type_train_0.tfrecords',
                                """CSV with training data""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_train', 3700,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 3,
                                """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt



NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000



def _generate_image_and_label_batch(image, label,min_queue_examples, batch_size, shuffle, train=True, num_preprocess_threads=5):

    if shuffle:
        images,  labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 2 * batch_size,
            seed=41,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 2 * batch_size)

    return images,  labels





def load_csv(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=str)
    images = data[:,2]
    labels = data[:,3]
    return images, labels




def read_and_decode(filename_queue):
    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
      features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'binary_segmentation': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)

      })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].



    left_image = tf.decode_raw(features['image_raw'], tf.uint8)
    bs = tf.decode_raw(features['binary_segmentation'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    left_image = tf.reshape(left_image, image_shape)

    bas_shape = tf.stack([height, width, 1])
    bs = tf.reshape(bs, bas_shape)



    image = tf.concat([left_image, bs], axis=2)
    # image = left_image

    image.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels])


    image = tf.cast(image, tf.float32)


    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.decode_raw(features['label'], tf.uint8)
    label_shape = tf.stack([height, width, 7])
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.float32)
    #
    # label_probe = label[:,:,14]
    # label_corr_0 = label[:,:,0] + label_probe
    # label_corr_0 = tf.expand_dims(label_corr_0,2)
    # label_corr_1 = label[:,:,1] - label_probe
    # label_corr_1 = tf.expand_dims(label_corr_1,2)
    #
    # label = tf.concat((label_corr_0, label_corr_1), axis=2)

    # label.set_shape([FLAGS.image_height, FLAGS.image_width])
    # label = tf.expand_dims(label,2)
    label.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.num_classes])
    label = tf.cast(label, tf.float32) # for binary segmentaiton

    return image, label







def random_rotate(image, label, max_angle=10):

    angle = tf.random_uniform([1, 1] ,minval=-max_angle, maxval=max_angle,dtype=tf.float32) / 360 * math.pi


    image = tf.contrib.image.rotate(image, angle[0][0])


    ones = tf.ones(shape=[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32), 1])

    zeros = tf.zeros(shape=[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32), 1])

    if(FLAGS.num_classes > 1):
        ones_ch =  tf.concat([ones, tf.tile(zeros,[1,1,FLAGS.num_classes-1])], axis=2)
    else:
        ones_ch =  tf.concat([ones], axis=2)


    label = label - ones_ch
    label = tf.contrib.image.rotate(label, angle[0][0])
    # label = tf.minimum(1.0, label)
    label = label + ones_ch


    return image, label


def random_horizontal_flip(image, label):

    flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32)

    image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)

    label = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_left_right(label), lambda: label)

    return image, label

def random_vertical_flip(image, label):

    flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32)

    image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)

    label = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_up_down(label), lambda: label)

    return image, label





def random_shift(image, label, max_shift):

    shift = tf.random_uniform([1, 2], minval=0, maxval=max_shift, dtype=tf.int32)
    print(image.shape)
    image = tf.image.crop_to_bounding_box(
                    image=image,
                    offset_height=shift[0][0],
                    offset_width= shift[0][1], target_height=tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32) - shift[0][0],
                    target_width=tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) - shift[0][1])

    image = tf.image.pad_to_bounding_box(
                    image=image,
                    offset_height=tf.abs(shift[0][0]),
                    offset_width=tf.abs(shift[0][1]),
                    target_height=tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),
                    target_width=tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32)
                )


    image.set_shape([
                    FLAGS.image_height * FLAGS.resize_factor,
                    FLAGS.image_width * FLAGS.resize_factor,
                    FLAGS.num_channels])

    ones = tf.ones(shape=[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32), 1])

    zeros = tf.zeros(shape=[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32), 1])

    if(FLAGS.num_classes > 1):
        ones_ch =  tf.concat([ones, tf.tile(zeros,[1,1,FLAGS.num_classes-1])], axis=2)
    else:
        ones_ch =  tf.concat([ones], axis=2)

    label = label - ones_ch
    label = tf.image.crop_to_bounding_box(
                    image=label,
                    offset_height=shift[0][0],
                    offset_width= shift[0][1], target_height=tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32) - shift[0][0],
                    target_width=tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) - shift[0][1])

    label = tf.image.pad_to_bounding_box(
                    image=label,
                    offset_height=shift[0][0],
                    offset_width=shift[0][1],
                    target_height=tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),
                    target_width=tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32))

    label = label + ones_ch
    # print(label.shape)
    return image, label




def distorted_inputs_files(filename, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  with tf.name_scope('input'):
      print("Loading distorted images")
    #   image_filenames, label_filenames = load_csv(filename)


  with tf.device('/cpu:0'):

        filename_queue = tf.train.string_input_producer([filename])

        image, label = read_and_decode(filename_queue)
        image = tf.image.resize_images(image,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ] )

        print(label.shape)
        label = tf.image.resize_images(label,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        print(label.shape)
        # if(FLAGS.data_augmentation):


        image, label = random_horizontal_flip(image, label)
        image, label = random_vertical_flip(image, label)
        image, label = random_shift(image, label, max_shift=100)
        image, label = random_rotate(image, label, max_angle=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25, seed=42)
        image = tf.image.random_brightness(image,max_delta=16. / 255., seed=42)

        image = tf.cast(image, dtype=tf.float32) / 255.0
            # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            # image = tf.image.random_brightness(image,max_delta=32. / 255.)

        mask = (1-label[:,:,0])
        mask = tf.expand_dims(mask, axis=2)
        image = image * tf.tile(mask,[1,1,4])
        image = tf.clip_by_value(image, 0.0, 1.0)


        #



        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                       min_fraction_of_examples_in_queue)
        print ('Filling queue with %d images before starting to train. '
                     'This will take a few minutes.' % min_queue_examples)

        return _generate_image_and_label_batch(image, label,
                                                     min_queue_examples, batch_size,
                                                     shuffle=False, num_preprocess_threads=int(8/FLAGS.num_gpus))



def main(unused_argv):

    with tf.Graph().as_default():
        images, labels = distorted_inputs_files(filename=FLAGS.file, batch_size=1)


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

            for i in range(0, 225*7):
            # while(True):
                start_time = time.time()
                images_s, labels_s= sess.run([images, labels])
                duration = time.time() - start_time
                print(images_s.shape)
                # print(images_s[0,:,:,0].shape)
                fig = plt.figure(figsize=(20, 10))

                plt.subplot(131)
                plt.imshow(images_s[0,:,:,0:3], vmin=0, vmax=1.0)
                plt.axis('off')
                #
                plt.subplot(132)
                plt.imshow(images_s[0,:,:,3], vmin=0, vmax=1.0)
                plt.axis('off')

                plt.subplot(133)
                plt.imshow(np.argmax(labels_s[0,:,:,:],2)*35)
                plt.axis('off')
                plt.savefig('tests/image' + str(i) + '.png',bbox_inches='tight')
                plt.close()


        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
  tf.app.run(main=main)
