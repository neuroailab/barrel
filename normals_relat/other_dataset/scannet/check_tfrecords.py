import tensorflow as tf
import numpy as np
from PIL import Image

tfrecords_filename = '/om/user/chengxuz/Data/one_world_dataset/scannet/tfrecs/normals/scannet0.tfrecord'
reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    #img_1d = np.fromstring(img_string, dtype=np.uint16)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    print(img_1d.shape)
    reconstructed_img = img_1d.reshape((480, 640, 3))
    
    reconstructed_images.append(reconstructed_img)

result = Image.fromarray(reconstructed_images[0])
result.save('normal_test.png')
