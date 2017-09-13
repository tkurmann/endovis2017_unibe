import tensorflow as tf


tf.app.flags.DEFINE_string('train_dir', 'model/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', 'model/eval/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('experiment', 'unet',
                                """experiment name""")




tf.app.flags.DEFINE_string('training_file', '../../data_export/tf/binary_train_0_all.tfrecords',
                                """training data""")
tf.app.flags.DEFINE_string('eval_file', '../../data_export/tf/binary_eval_0_all.tfrecords',
                            """CSV with training data""")



tf.app.flags.DEFINE_integer('seed', 42,
                                """Number of examples to run.""")




tf.app.flags.DEFINE_integer('eval_examples', 1000,
                                """Number of examples to run.""")
tf.app.flags.DEFINE_integer('train_batch_size',4,
                                """training batch size""")

tf.app.flags.DEFINE_integer('eval_batch_size',1,
                                """training batch size""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                                """Number of batches to run.""")
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
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                                """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_train', 1575,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 50,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_integer('unet_layers', 5,
                                """Number of channels in input image.""")
tf.app.flags.DEFINE_integer('unet_features_root', 32,
                                """Number of channels in input image.""")
tf.app.flags.DEFINE_boolean('unet_dualconv', True,
                                """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('weighted_loss', False,
                                """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('unet_kernel', 3,
                                """Kernel size for convolutions""")

tf.app.flags.DEFINE_boolean('unet_shortskip', False,
                                """Train the model using fp16.""")

tf.app.flags.DEFINE_float('class_weights', 1.0,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_string('crossval_run', 0,
                                """experiment name""")
tf.app.flags.DEFINE_boolean('data_augmentation', True,
                                """Train the model using fp16.""")
tf.app.flags.DEFINE_integer('data_augmentation_level', 9,
                                """level of data augmentation""")

tf.app.flags.DEFINE_float('eval_gpu_fraction', 0.95,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_float('train_gpu_fraction', 0.95,
                                """Whether to log device placement.""")
tf.app.flags.DEFINE_string('loss_type', 'crossentropy',
                           """which loss to use, cs or dice""")
