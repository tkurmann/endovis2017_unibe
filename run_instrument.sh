#!bin/bash

# sudo apt-get install -y git
# sudo pip3 install cython joblib
# sudo pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git



instrument=$1
remove_probe=$2


echo "instrument_$instrument"

cd data_export

python3 create_instrument_set.py --split="train" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/train/"

python3 create_instrument_set.py --split="eval" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/eval/"
#
python3 create_instrument_set.py --split="test" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/test/" --binary_results_start=225

cd ../src/instrument_type_segmentation

python3 train.py --experiment="type_$instrument" --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True  --training_file="../../data_export/tf/type_train_$instrument.tfrecords" --eval_file="../../data_export/tf/type_eval_$instrument.tfrecords" --max_steps=50000 --initial_learning_rate=1e-4 --resize_factor=0.3 

python3 inference.py --inference_file="../../data_export/tf/type_test_$instrument.tfrecords"  --output_folder="results/$instrument/test/" --models_folder="model/train/type_$instrument" --frame_start_number=225 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

