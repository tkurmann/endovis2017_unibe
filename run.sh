#!bin/bash

# sudo apt-get install -y git
# sudo pip3 install cython joblib
# sudo pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git



instrument=$1
remove_probe=$2


echo "binary_$instrument"

cd data_export
python3 create_binary_set.py --split="train" --removeprobe=$remove_probe  --leaveout=$instrument
python3 create_binary_set.py --split="eval" --removeprobe=$remove_probe  --leaveout=$instrument
python3 create_binary_set.py --split="test" --removeprobe=$remove_probe  --leaveout=$instrument


cd ../src/binary_segmentation
python3 train.py --experiment="binary_$instrument" --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True  --training_file="../../data_export/tf/binary_train_$instrument.tfrecords" --eval_file="../../data_export/tf/binary_eval_$instrument.tfrecords" --max_steps=50000 --resize_factor=0.3 --initial_learning_rate=1e-4 --loss_type="dice"
#
python3 inference.py --inference_file="../../data_export/tf/binary_test_$instrument.tfrecords"  --output_folder="results/$instrument/test/" --models_folder="model/train/binary_$instrument" --frame_start_number=225 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

python3 inference.py --inference_file="../../data_export/tf/binary_train_$instrument.tfrecords" --output_folder="results/$instrument/train/" --models_folder="model/train/binary_$instrument" --frame_start_number=0 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

python3 inference.py --inference_file="../../data_export/tf/binary_eval_$instrument.tfrecords" --output_folder="results/$instrument/eval/" --models_folder="model/train/binary_$instrument" --frame_start_number=0 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

cd ../../data_export

python3 create_parts_set.py --split="train" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/train/"

python3 create_parts_set.py --split="eval" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/eval/"
#
python3 create_parts_set.py --split="test" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/test/" --binary_results_start=225


cd ../src/instrument_part_segmentation

python3 train.py --experiment="parts_$instrument" --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True  --training_file="../../data_export/tf/parts_train_$instrument.tfrecords" --eval_file="../../data_export/tf/parts_eval_$instrument.tfrecords" --max_steps=50000 --initial_learning_rate=1e-4 --resize_factor=0.3 --loss_type="dice" --seed=42


python3 inference.py --inference_file="../../data_export/tf/parts_test_$instrument.tfrecords"  --output_folder="results/$instrument/test/" --models_folder="model/train/parts_$instrument" --frame_start_number=225 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

python3 inference.py --inference_file="../../data_export/tf/parts_train_$instrument.tfrecords" --output_folder="results/$instrument/train/" --models_folder="model/train/parts_$instrument" --frame_start_number=0 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3

python3 inference.py --inference_file="../../data_export/tf/parts_eval_$instrument.tfrecords" --output_folder="results/$instrument/eval/" --models_folder="model/train/parts_$instrument" --frame_start_number=0 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3


cd ../../data_export

python3 create_instrument_set.py --split="train" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/train/"

python3 create_instrument_set.py --split="eval" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/eval/"
#
python3 create_instrument_set.py --split="test" --leaveout=$instrument --removeprobe=True --binary_results_folder="../src/binary_segmentation/results/$instrument/test/" --binary_results_start=225

cd ../src/instrument_type_segmentation

python3 train.py --experiment="type_$instrument" --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True  --training_file="../../data_export/tf/type_train_$instrument.tfrecords" --eval_file="../../data_export/tf/type_eval_$instrument.tfrecords" --max_steps=50000 --initial_learning_rate=1e-4 --resize_factor=0.3 

python3 inference.py --inference_file="../../data_export/tf/type_test_$instrument.tfrecords"  --output_folder="results/$instrument/test/" --models_folder="model/train/type_$instrument" --frame_start_number=225 --unet_layers=6 --unet_features_root=32 --unet_kernel=3 --data_augmentation_level=9 --unet_shortskip=True --seed=42 --resize_factor=0.3









