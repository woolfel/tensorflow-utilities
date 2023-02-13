# tensorflow-utilities

Colllection of utilities for Tensorflow 2 and Keras. 

## Install & Setup

If you are new to tensorflow, there's a handy conda configuration for setting up a virtual environment. There's one for Windows and one for MacOS. After you clone the repo, use conda to create the environment. If you're new to conda, there are youtube tutorials on conda.


## Convert Keras H5 model to JSON

Sometimes you might want to convert a Keras Sequential model to human readable format or load it in another language. Tensorflow JS used to provide a way to do this, but it is deprecated as of January 2023. I tried to use it, but it's basically broken until Google's team decides to fix it.

Example usage

python convert_keras_h5_to_json.py my_model.h5 my_mode.json

## Profile inference and save the results

Script demonstrates how to profile inference with Tensorflow and Keras. The script will load your model, a tensorflow dataset, test image and output file name. If you're using tensorflow dataset, it should work "as is". If your dataset is not a standard tensorflow dataset, you can convert it to tensorflow dataset first.

python tfds_profile_activation my_mode.h5 cifar10 test_09590 profile_result.json
