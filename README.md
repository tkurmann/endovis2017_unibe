# endovis2017_unibe
Code to generate the our result of the 2017 Endovis Robotic Instrument Segmentation Challenge

In order to run the code you need to install a couple of dependancies:
- Numpy
- Tensorflow 1.2.1
- Cython
- Joblib
- Scipy

Please note that a GPU with at least 8GB of memory is required to train the models. Training time for all 24 required models will take roughly 8 days on a single GPU.
The dataset needs to be manually downloaded to the data_export/data_train and data_export/data_test folder.

To train all models please run sh run_all.sh

