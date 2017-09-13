from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import h5py
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import math
import settings

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
_IMAGE_DTYPES = set(
[dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])


FLAGS = tf.app.flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


seed = 42


images_std = np.array([ 65.01658218,  48.48053989 , 53.29497727])
images_mean = np.array([ 99.45559037,  63.53085635 , 70.73308415])



def _generate_image_and_label_batch(image, label,min_queue_examples, batch_size, shuffle, train=True, num_preprocess_threads=5):

    if shuffle:
        images,  labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 2 * batch_size,
            seed=FLAGS.seed,
            min_after_dequeue=min_queue_examples,
            enqueue_many=False)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 2 * batch_size,
            enqueue_many=False)

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
           'binary_segmentation': tf.FixedLenFeature([], tf.string),
           'image_raw': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([], tf.string)
     })

   # Convert from a scalar string tensor (whose single string has
   # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
   # [mnist.IMAGE_PIXELS].



    left_image = tf.decode_raw(features['image_raw'], tf.uint8)
    bs = tf.decode_raw(features['binary_segmentation'], tf.uint8)
    #right_image = tf.decode_raw(features['right_image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    left_image = tf.reshape(left_image, image_shape)
    #right_image = tf.reshape(right_image, image_shape)

    bas_shape = tf.stack([height, width, 1])
    bs = tf.reshape(bs, bas_shape)
    bs = (255-tf.cast(bs, tf.float32))/255

    if(FLAGS.num_channels == 6):
        image = tf.concat([left_image, right_image], axis=2)
    else:
        image = left_image
       # image = left_image



    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.decode_raw(features['label'], tf.uint8)
    label_shape = tf.stack([height, width, 7])
    label = tf.reshape(label, label_shape)
    label = tf.cast(label, tf.float32)

    # label.set_shape([FLAGS.image_height, FLAGS.image_width])
    # label = tf.expand_dims(label,2)
    print(label)
    label.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.num_classes])


    image = tf.cast(image, tf.float32)








    return image, label, bs









def random_rotate(image, label, max_angle=10):

    angle = tf.random_uniform([1, 1] ,minval=-max_angle, maxval=max_angle,dtype=tf.float32, seed=FLAGS.seed) / 360 * math.pi


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

    flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=FLAGS.seed)

    image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)

    label = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_left_right(label), lambda: label)

    return image, label

def random_vertical_flip(image, label):

    flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=FLAGS.seed)

    image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)

    label = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_up_down(label), lambda: label)

    return image, label





def random_shift(image, label, max_shift):

    shift = tf.random_uniform([1, 2], minval=0, maxval=max_shift, dtype=tf.int32, seed=FLAGS.seed)
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

        image, label, binary_segmentation = read_and_decode(filename_queue)





        image = tf.image.resize_images(image,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ] )


        label = tf.image.resize_images(label,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

        binary_segmentation=tf.image.resize_images(binary_segmentation,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

        image = tf.cast(image, dtype=tf.float32) / 255.0
        # image = tf.image.per_image_standardization(image)

        rv = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=FLAGS.seed)

        # image = tf.cond(rv[0,0] > 0.5, lambda: tf.concat([image, binary_segmentation], axis=2), lambda: tf.concat([image, tf.expand_dims(label[:,:,0], axis=2)], axis=2))
        #mask = (1-label[:,:,0])
        #mask = tf.expand_dims(mask, axis=2)
        #image = image * tf.tile(mask,[1,1,3])


        image = tf.concat([image, tf.expand_dims(label[:,:,0], axis=2)], axis=2)




        if(FLAGS.data_augmentation_level == 1):
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5, seed=FLAGS.seed)
        if(FLAGS.data_augmentation_level == 2):
            image = tf.image.random_brightness(image,max_delta=32. / 255., seed=FLAGS.seed)
        if(FLAGS.data_augmentation_level == 3):
            image = tf.image.random_hue(image, max_delta=0.2, seed=FLAGS.seed)
        if(FLAGS.data_augmentation_level == 4):
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=FLAGS.seed)

        if(FLAGS.data_augmentation_level == 5):
            image, label = random_horizontal_flip(image, label)
        if(FLAGS.data_augmentation_level == 6):
            image, label = random_vertical_flip(image, label)
        if(FLAGS.data_augmentation_level == 7):
            image, label = random_shift(image, label, max_shift=100)
        if(FLAGS.data_augmentation_level == 8):
            image, label = random_rotate(image, label, max_angle=30)

        if(FLAGS.data_augmentation_level == 9):
            image, label = random_horizontal_flip(image, label)
            image, label = random_vertical_flip(image, label)
            image, label = random_shift(image, label, max_shift=100)
            image, label = random_rotate(image, label, max_angle=30)
            # image = tf.image.random_contrast(image, lower=0.75, upper=1.25, seed=FLAGS.seed)
            # image = tf.image.random_brightness(image,max_delta=16. / 255., seed=FLAGS.seed)
            # # image = tf.image.random_hue(image, max_delta=0.2, seed=FLAGS.seed)
            # image = tf.image.random_saturation(image, lower=0.75, upper=1.25, seed=FLAGS.seed)


        image = tf.concat([image[:,:,0:3], tf.expand_dims(label[:,:,0], axis=2)], axis=2)
            # image = tf.clip_by_value(image, 0.0, 1.0)

        #



        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                       min_fraction_of_examples_in_queue)
        print ('Filling queue with %d images before starting to train. '
                     'This will take a few minutes.' % min_queue_examples)

        return _generate_image_and_label_batch(image, label,
                                                     min_queue_examples, batch_size,
                                                     shuffle=True, num_preprocess_threads=int(8/FLAGS.num_gpus))


def eval_inputs_files(filename, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  with tf.name_scope('input'):
    print("Loading image filenames")
    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    num_records = 0
    for record in tf.python_io.tf_record_iterator(filename, options=options):
        num_records += 1


    with tf.device('/cpu:0'):
        filename_queue = tf.train.string_input_producer([filename])
        image, label, binary_segmentation = read_and_decode(filename_queue)
        image = tf.image.resize_images(image,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ] )

        label = tf.image.resize_images(label,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )

        binary_segmentation=tf.image.resize_images(binary_segmentation,[tf.cast(FLAGS.image_height * FLAGS.resize_factor,tf.int32),tf.cast(FLAGS.image_width * FLAGS.resize_factor,tf.int32) ], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )


        min_fraction_of_examples_in_queue = 0.25
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                           min_fraction_of_examples_in_queue)
        print ('Filling queue with %d images fo evaluation. '
                         'This will take a few minutes.' % min_queue_examples)

        image = tf.cast(image, dtype=tf.float32) / 255.0
        # image = tf.image.per_image_standardization(image)
        #mask = (1-label[:,:,0])
        #mask = tf.expand_dims(mask, axis=2)
        #image = image * tf.tile(mask,[1,1,3])

        image = tf.concat([image, binary_segmentation], axis=2)


        images, labels = _generate_image_and_label_batch(image, label,
                                                         min_queue_examples, batch_size,
                                                         shuffle=False, num_preprocess_threads=4)

        return images, labels , num_records
