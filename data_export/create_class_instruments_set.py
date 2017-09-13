import tensorflow as tf
from scipy import misc
import json
import sys
import os
import glob
import numpy as np
from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import uuid
import random


tf.app.flags.DEFINE_string('split', 'train',"""train, eval or test split""")
tf.app.flags.DEFINE_integer('leaveout', 0,
                                """leave out which instrument""")
tf.app.flags.DEFINE_boolean('removeprobe', True,
                                """Remove probe in """)

tf.app.flags.DEFINE_string('binary_results_folder', '',"""train, eval or test split""")
tf.app.flags.DEFINE_integer('binary_results_start', 0,
                                """leave out which instrument""")
FLAGS = tf.app.flags.FLAGS


TRAIN_DATA_BASE_DIR = "data_train/"
TEST_DATA_BASE_DIR = "data_test/"

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 1280


def crop_frame(image):

    im_shape = image.shape

    min_x = 320
    min_y = 0
    max_x = min_x+IMAGE_WIDTH
    max_y = min_y+1080 # limited due to height...
    if(len(image.shape) == 3):
        image_new = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, im_shape[2]))
        image_new[100:100+1080, :, :] = image[:,min_x:max_x,:]
    else:
        image_new = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        image_new[100:100+1080, :] = image[:,min_x:max_x]



    return image_new

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images,  labels,  name):
  """Converts a dataset to tfrecords."""
  # images = data_set.images
  # labels = data_set.labels
  num_examples = images.shape[0]

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  try:
      depth = images.shape[3]
  except:
      depth = 1

  filename = os.path.join("tf/", name + '.tfrecords')
  print('Writing', filename)
  options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
  writer = tf.python_io.TFRecordWriter(filename, options=options)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    label_raw = labels[index]
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()



class FileSet():
    def __init__(self, left_image_file, right_image_file, ground_truth_files=[], ground_truth_instrument_type=[]):
        self.left_image_file = left_image_file
        self.right_image_file = right_image_file
        self.ground_truth_files = ground_truth_files
        self.ground_truth_instrument_type = ground_truth_instrument_type


    def load_all(self):
        print("loading al")

    # def __str__(self):
    #     string = "Image File"+self.image_file
    #     for gt_file in self.ground_truth_files:
    #         string += ";GT:" + gt_file
    #     return string

class InstrumentSet():

    def __init__(self, left_image_folder, right_image_folder, ground_truth_folders, ground_truth_instrument_type):
        self.left_image_folder = left_image_folder
        self.right_image_folder = right_image_folder
        self.ground_truth_folders = ground_truth_folders
        self.ground_truth_instrument_type = ground_truth_instrument_type
        self.filesets = []


    def gather_data_files(self):
        left_files = sorted(glob.glob(self.left_image_folder+"/*.png"))
        right_files = sorted(glob.glob(self.right_image_folder+"/*.png"))
        filesets = []


        for f in range(len(left_files)):
            left_file = left_files[f]
            right_file = right_files[f]
            filesets.append(FileSet(left_image_file=left_file, right_image_file=right_file, ground_truth_files=[], ground_truth_instrument_type=[]))

        self.filesets = filesets

    def gather_and_map_ground_truths(self):

        for g in range(len(self.ground_truth_folders)):
            gt_folder = self.ground_truth_folders[g]
            instrument_type = self.ground_truth_instrument_type[g]

            files = sorted(glob.glob(gt_folder+"/*.png"))

            for file in files:
                for f in range(len(self.filesets)):
                    fileset = self.filesets[f]
                    if(os.path.basename(fileset.left_image_file) == os.path.basename(file)):
                        fileset.ground_truth_files.append(file)
                        fileset.ground_truth_instrument_type.append(instrument_type)
                        break;


    def generate_parts_instruments_labels(self, instrument_parts, instrument_types):

        num_classes = (len(instrument_parts)-1)*len(instrument_types)+1
        print(num_classes)
        left_image_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 3 ), dtype=np.uint8)

        right_image_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 3 ), dtype=np.uint8)

        gt_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, num_classes), dtype=np.uint8)

        for f in range(len(self.filesets)):
        # for f in range(1):
            print(fileset.left_image_file)
            fileset = self.filesets[f]
            left_image = misc.imread(fileset.left_image_file)
            left_image = crop_frame(left_image)
            left_image_data[f,:,:,:] = left_image

            right_image = misc.imread(fileset.right_image_file)
            right_image = crop_frame(right_image)
            right_image_data[f,:,:,:] = right_image

            gt_file = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))


            for g in range(len(fileset.ground_truth_files)):
                gt = misc.imread(fileset.ground_truth_files[g], flatten=True) / 10# scale from 0..4 instead of 0...40
                print(gt.shape)
                instrument_type = fileset.ground_truth_instrument_type[g]
                print(instrument_type)
                instrument_type_id = instrument_types[instrument_type] -1 # correct for 1 start
                print(instrument_type_id)
                gt = crop_frame(gt)
                print(np.max(gt))

                gt[gt < 0.9] = 'nan'
                gt[~np.isnan(gt)] = (gt[~np.isnan(gt)] -1) + 1 + instrument_type_id*(len(instrument_parts)-1)

                gt[np.isnan(gt)] = 0
                idx_not_0 = np.where(gt!=0)

                gt_file[idx_not_0] = gt[idx_not_0]

            # print("Gt file")
            # print(gt_file.shape)
            # create one "channel" per label for crossentropy loss
            gt_exp = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, num_classes])
            xv, yv = np.meshgrid(np.linspace(0, IMAGE_WIDTH-1, IMAGE_WIDTH),np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT))
            # xv and yv contain the x and y indexes. slice_reference contains values from 0 to 3(so the channel indexes)
            gt_exp[yv.reshape(-1).astype(np.int32), xv.reshape(-1).astype(np.int32), gt_file.reshape(-1).astype(np.int32) ] = 1
            gt = gt_exp.astype(np.uint8)
            # print(gt.shape)
            gt_data[f,:,:,:] = gt
            print(np.mean(gt[:,:,0]))


        return left_image_data, right_image_data,  gt_data


    def generate_binary_labels(self, instrument_parts, instrument_types):

        num_classes = 2 # binary map
        num_instruments = 8
        num_parts = 4
        print(num_classes)
        left_image_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 3 ), dtype=np.uint8)

        cropped_image_data = np.zeros((2*len(self.filesets)*len(self.filesets[0].ground_truth_files), 256, 256, 3 ), dtype=np.uint8)



        gt_data = np.zeros((2*len(self.filesets)*len(self.filesets[0].ground_truth_files)), dtype=np.uint8)

        # right_image_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 3 ), dtype=np.uint8)

        binary_gt_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 2), dtype=np.uint8)

        instrument_gt_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, 7), dtype=np.uint8)
        #
        # part_gt_data = np.zeros((len(self.filesets), IMAGE_HEIGHT, IMAGE_WIDTH, num_parts), dtype=np.uint8)

        num_images = 0

        for f in range(len(self.filesets)):
        # for f in range(5):

            fileset = self.filesets[f]
            print(fileset.left_image_file)
            left_image = misc.imread(fileset.left_image_file)
            left_image = crop_frame(left_image)
            left_image_data[f,:,:,:] = left_image

            # right_image = misc.imread(fileset.right_image_file)
            # right_image = crop_frame(right_image)
            # right_image_data[f,:,:,:] = right_image

            gt_binary_file = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
            gt_instrument_file = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
            gt_part_file = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

            for g in range(len(fileset.ground_truth_files)):
                gt = misc.imread(fileset.ground_truth_files[g], flatten=True) / 10# scale from 0..4 instead of 0...40
                print(gt.shape)
                instrument_type = fileset.ground_truth_instrument_type[g]
                print(instrument_type)
                instrument_type_id = instrument_types[instrument_type]
                print(instrument_type_id)
                gt = crop_frame(gt)
                print(np.max(gt))

                if((instrument_type_id == 7) and FLAGS.removeprobe): # If its other it isnt an instrument....
                    print("remove_probe")
                    gt = np.zeros(gt.shape)



                gt_binary_file += gt

                # idx_not_0 = np.where(gt!=0)
                # gt_tmp = np.copy(gt)
                #
                # gt_part_file[idx_not_0] = gt_tmp[idx_not_0]

                #
                #
                gt[gt < 2.5] = 'nan'
                gt[~np.isnan(gt)] = instrument_type_id

                gt[np.isnan(gt)] = 0
                idx_not_0 = np.where(gt!=0)

                gt_instrument_file[idx_not_0] = gt[idx_not_0]




            blur_radius = 1.0
            gt_instrument_file_filt = ndimage.gaussian_filter(gt_instrument_file, blur_radius)
            labeled, nr_objects = ndimage.label(gt_instrument_file_filt > 0)
            print(labeled.shape)
            print("Nr Objects:"+str(nr_objects))

            print(cropped_image_data.shape)
            for l in range(nr_objects):
                print("unique")
                print(np.unique(gt_instrument_file[labeled==(l+1)]))
                if(len(np.unique(gt_instrument_file[labeled==(l+1)])) == 2):
                    labels = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH))
                    labels[labeled==(l+1)] = 1



                    labels = np.reshape(labels,(IMAGE_HEIGHT,IMAGE_WIDTH))
                    labels = np.clip(labels, 0, 1)
                    print(np.sum(labels))

                    xv, yv = np.meshgrid(np.linspace(0, IMAGE_WIDTH-1, IMAGE_WIDTH),np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT))


                    lxv = labels*xv
                    lxv = lxv[np.nonzero(lxv)]

                    lyv = labels*yv
                    lyv = lyv[np.nonzero(lyv)]

                    min_x = np.min(lxv).astype(np.int32)
                    max_x = np.max(lxv).astype(np.int32)

                    min_y = np.min(lyv).astype(np.int32)
                    max_y = np.max(lyv).astype(np.int32)

                    len_x = (max_x - min_x)
                    len_y = (max_y - min_y)

                    if(len_x< 128):
                        size_increase = (128-len_x)/2
                        min_x -= size_increase
                        max_x += size_increase
                        min_x=np.clip(min_x, 0, 1280)
                        max_x=np.clip(max_x, 0, 1280)

                    if(len_y< 128):
                        size_increase = (128-len_y)/2
                        min_y -= size_increase
                        max_y += size_increase
                        min_y=np.clip(min_y, 0, 1280)
                        max_y=np.clip(max_y, 0, 1280)


                    max_x = int(max_x)
                    max_y = int(max_y)
                    min_x = int(min_x)
                    min_y = int(min_y)
                    cropped_image = left_image[min_y:max_y, min_x:max_x,:]

                    largest_side = np.max(cropped_image.shape)
                    factor = largest_side/256


                    cropped_image = misc.imresize(cropped_image, size=(np.round(cropped_image.shape[0]/factor).astype(np.int32),np.round(cropped_image.shape[1]/factor).astype(np.int32)))

                    cropped_image_256 = np.zeros((256,256,3))

                    x_size = cropped_image.shape[0]
                    y_size = cropped_image.shape[1]

                    cropped_image_256[128-np.floor(x_size/2).astype(np.int32):128+np.ceil(x_size/2).astype(np.int32), 128-np.floor(y_size/2).astype(np.int32):128+np.ceil(y_size/2).astype(np.int32), :] = cropped_image

                    # misc.imsave("cropped/"+str(uuid.uuid4())+".png", cropped_image_256)
                    cropped_image_data[num_images, : , :, : ] = cropped_image_256
                    gt_data[num_images] = np.unique(gt_instrument_file[labeled==(l+1)])[1]
                    num_images = num_images + 1
                    print(gt_data.shape)






            # # binary_gt_data[f,:,:,1] = np.clip(gt_binary_file, 0, 1) # instrument
            # # binary_gt_data[f,:,:,0] = 1 - binary_gt_data[f,:,:,1] # background
            #
            # # instruments tyype
            # gt_exp = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, num_instruments])
            # xv, yv = np.meshgrid(np.linspace(0, IMAGE_WIDTH-1, IMAGE_WIDTH),np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT))
            # # xv and yv contain the x and y indexes. slice_reference contains values from 0 to 3(so the channel indexes)
            # gt_exp[yv.reshape(-1).astype(np.int32), xv.reshape(-1).astype(np.int32), gt_instrument_file.reshape(-1).astype(np.int32) ] = 1
            # gt_instrument_file = gt_exp.astype(np.uint8)
            # # print(gt.shape)
            # instrument_gt_data[f,:,:,:] = gt_instrument_file
            # #
            #
            # instruments parts
            # gt_part_file = np.clip(gt_part_file, 0 , num_parts-1)
            # gt_exp = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, num_parts])
            # xv, yv = np.meshgrid(np.linspace(0, IMAGE_WIDTH-1, IMAGE_WIDTH),np.linspace(0, IMAGE_HEIGHT-1, IMAGE_HEIGHT))
            # # xv and yv contain the x and y indexes. slice_reference contains values from 0 to 3(so the channel indexes)
            # gt_exp[yv.reshape(-1).astype(np.int32), xv.reshape(-1).astype(np.int32), gt_part_file.reshape(-1).astype(np.int32) ] = 1
            # gt_part_file = gt_exp.astype(np.uint8)
            # # print(gt.shape)
            # part_gt_data[f,:,:,:] = gt_part_file



        # gt_data = np.concatenate([binary_gt_data, instrument_gt_data, part_gt_data], axis=3)
        # print(gt_data.shape)
        cropped_image_data = cropped_image_data[0:num_images, : , : , :]
        gt_data = gt_data[0:num_images]
        print(gt_data.shape)
        return cropped_image_data,  gt_data



def main(unused_argv):


    instrument_parts_file = 'mappings.json'
    instrument_type_file = 'instrument_type_mapping.json'

    with open(TRAIN_DATA_BASE_DIR+instrument_parts_file) as data_file:
        instrument_parts = json.load(data_file)

    with open(TRAIN_DATA_BASE_DIR+instrument_type_file) as data_file:
        instrument_types = json.load(data_file)

    training_list = []
    test_list = []

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_1/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_1/right_frames",[TRAIN_DATA_BASE_DIR+"instrument_dataset_1/ground_truth/Maryland_Bipolar_Forceps_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_1/ground_truth/Other_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_1/ground_truth/Left_Prograsp_Forceps_labels", TRAIN_DATA_BASE_DIR+"instrument_dataset_1/ground_truth/Right_Prograsp_Forceps_labels" ],["Bipolar Forceps","Other","Prograsp Forceps","Prograsp Forceps" ]))

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_2/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_2/right_frames", [TRAIN_DATA_BASE_DIR+"instrument_dataset_2/ground_truth/Left_Prograsp_Forceps_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_2/ground_truth/Other_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_2/ground_truth/Right_Prograsp_Forceps_labels" ],["Prograsp Forceps","Other","Prograsp Forceps" ]))

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_3/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_3/right_frames",[TRAIN_DATA_BASE_DIR+"instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_3/ground_truth/Right_Large_Needle_Driver_labels" ],["Large Needle Driver","Large Needle Driver"]))
    #
    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_4/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_4/right_frames",[TRAIN_DATA_BASE_DIR+"instrument_dataset_4/ground_truth/Large_Needle_Driver_Left_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_4/ground_truth/Large_Needle_Driver_Right_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_4/ground_truth/Prograsp_Forceps_labels" ],["Large Needle Driver","Large Needle Driver","Prograsp Forceps" ]))

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_5/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_5/right_frames",[TRAIN_DATA_BASE_DIR+"instrument_dataset_5/ground_truth/Bipolar_Forceps_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_5/ground_truth/Grasping_Retractor_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_5/ground_truth/Vessel_Sealer_labels" ],["Bipolar Forceps2","Grasping Retractor","Vessel Sealer" ]))

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_6/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_6/right_frames",
    [TRAIN_DATA_BASE_DIR+"instrument_dataset_6/ground_truth/Left_Large_Needle_Driver_labels",
    TRAIN_DATA_BASE_DIR+"instrument_dataset_6/ground_truth/Monopolar_Curved_Scissors_labels",
    TRAIN_DATA_BASE_DIR+"instrument_dataset_6/ground_truth/Prograsp_Forceps", TRAIN_DATA_BASE_DIR+"instrument_dataset_6/ground_truth/Right_Large_Needle_Driver_labels" ]
    ,["Large Needle Driver","Monopolar Curved Scissors","Prograsp Forceps", "Large Needle Driver" ]))

    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_7/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_7/right_frames", [TRAIN_DATA_BASE_DIR+"instrument_dataset_7/ground_truth/Left_Bipolar_Forceps",TRAIN_DATA_BASE_DIR+"instrument_dataset_7/ground_truth/Right_Vessel_Sealer" ],["Bipolar Forceps2","Vessel Sealer"]))


    training_list.append(InstrumentSet(TRAIN_DATA_BASE_DIR+"instrument_dataset_8/left_frames",TRAIN_DATA_BASE_DIR+"instrument_dataset_8/right_frames", [TRAIN_DATA_BASE_DIR+"instrument_dataset_8/ground_truth/Bipolar_Forceps_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_8/ground_truth/Left_Grasping_Retractor_labels",TRAIN_DATA_BASE_DIR+"instrument_dataset_8/ground_truth/Monopolar_Curved_Scissors_labels", TRAIN_DATA_BASE_DIR+"instrument_dataset_8/ground_truth/Right_Grasping_Retractor_labels" ],["Bipolar Forceps","Grasping Retractor", "Monopolar Curved Scissors", "Grasping Retractor" ]))




    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_1/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_1/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_2/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_2/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_3/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_3/right_frames",[],[]))
    #
    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_4/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_4/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_5/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_5/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_6/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_6/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_7/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_7/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_8/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_8/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_9/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_9/right_frames",[],[]))

    test_list.append(InstrumentSet(TEST_DATA_BASE_DIR+"instrument_dataset_10/left_frames",TEST_DATA_BASE_DIR+"instrument_dataset_10/right_frames",[],[]))

    instrument_idx = list(range(len(training_list)))
    leave_out = FLAGS.leaveout

    if(FLAGS.split == "train"):

        set = np.array(training_list[:leave_out]+training_list[leave_out+1:])

    elif(FLAGS.split == "eval"):
        set = []
        set.append(training_list[leave_out])
        set = np.asarray(set)

    elif(FLAGS.split == "test"):
        set = []
        set.append(test_list[leave_out])
        set = np.asarray(set)



    images = np.empty([0,256,256, 3]).astype(np.uint8)
    labels = np.empty([0]).astype(np.uint8)

    bs_file = FLAGS.binary_results_start

    for instrument in set:
        instrument.gather_data_files()
        instrument.gather_and_map_ground_truths()
        image_data,  gt_data =  instrument.generate_binary_labels(instrument_parts, instrument_types)

        print(gt_data.shape)
        images = np.vstack((images, image_data))
        labels = np.append(labels, gt_data)
        # labels = np.vstack((labels, gt_data))

        print(images.shape)
        print(labels.shape)

    print(labels)


    if(FLAGS.split == "train"):
        historgram = np.bincount(labels.astype(np.int64))
        print(historgram)
        max_class = np.max(historgram)

        print(max_class)
        labels_balanced = np.empty([0]).astype(np.uint8)
        images_balanced = np.empty([0,256,256, 3]).astype(np.uint8)

        for c in range(1,9):
            idx = np.squeeze(np.asarray(np.where(labels==c)))
            print(idx.shape)
            if(idx.size != 0):
                if(idx.size == max_class):
                    print("skipping max class")
                    labels_balanced = np.append(labels_balanced,labels[idx])
                    images_balanced = np.vstack((images_balanced, np.squeeze(images[idx])))

                else:
                    print(idx.size)
                    missing = max_class - idx.size
                    print(missing)
                    idx_new =([random.choice(idx) for _ in range(missing)])
                    # idx_new = random.choices(idx, k=missing)
                    print(idx_new)
                    idx = np.append(idx, idx_new)
                    print(idx.shape)
                    labels_balanced = np.append(labels_balanced,labels[idx])
                    print(images[idx,:,:,:].shape)

                    images_balanced = np.vstack((images_balanced, np.squeeze(images[idx])))

    else:
        labels_balanced = labels
        images_balanced = images
    #

    filename_list = []
    for i in range(len(labels_balanced)):
        gt = labels_balanced[i]
        image = images_balanced[i,:,:,:]
        filename = "cropped/"+str(gt)+"_"+str(uuid.uuid4())+".png"
        misc.imsave(filename, image)
        filename_list.append(filename)

    # output = np.hstack((filename_list,labels_balanced.astype(np.uint8)))
    # print(output)


    np.savetxt("filename_"+FLAGS.split+".txt", filename_list,  fmt="%s")
    np.savetxt("label_"+FLAGS.split+".txt", labels_balanced.astype(np.uint8))

    historgram = np.bincount(labels_balanced.astype(np.int64))
    print(historgram)
    # n.savetxt("labels.txt", labels)
    # convert_to(images_balanced, labels_balanced, "class_"+FLAGS.split+"_"+str(leave_out))



if __name__ == '__main__':

  tf.app.run(main=main)
