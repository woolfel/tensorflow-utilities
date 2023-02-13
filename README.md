# tensorflow-utilities

Colllection of utilities for Tensorflow 2 and Keras. 

## Convert Keras H5 model to JSON

Sometimes you might want to convert a Keras Sequential model to human readable format or load it in another language. Tensorflow JS used to provide a way to do this, but it is deprecated as of January 2023. I tried to use it, but it's basically broken until Google's team decides to fix it.

Example usage

python convert_keras_h5_to_json.py my_model.h5 my_mode.json

## Profile inference and save the results

