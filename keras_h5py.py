import h5py
import numpy as np
from keras.applications import resnet50
import datetime

path = ['data/bathroom_02_keras.h5', 'data/bedroom_04_keras.h5', "data/kitchen_02_keras.h5", "data/living_room_08_keras.h5"]

# Use pretrained resnet50 from Keras with avg pooling
resnet_trained = resnet50.ResNet50(include_top=False, weights = 'imagenet', pooling='avg', input_shape = (300, 400, 3))
# Freeze all layers
for layer in resnet_trained.layers:
    layer.trainable = False


for p in path:
    h5_file = h5py.File(p, 'r+')
    features = []
    x = h5_file['observation'][:]
    try:
        del h5_file['resnet_feature']
    except KeyError:
        pass
    try:
        del h5_file['feature_date']
    except KeyError:
        pass
    x_new = np.asarray(x)
    x_new = resnet50.preprocess_input(x_new)
    outputs = resnet_trained.predict(x_new)
    outputs = outputs[:,np.newaxis,:]
    print(outputs.shape)
    h5_file.create_dataset('feature_date', data=[datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("ascii", "ignore")])
    h5_file.create_dataset('resnet_feature', data=outputs)
    h5_file.close()
