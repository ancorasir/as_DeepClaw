import tensorflow as tf 
import numpy as np 
from PIL import Image
import pandas as pd
import os

# src_folder = ''
src_folder = '/home/ancora-sirlab/arcade_claw_test/2017-03-27 10:35:20'
dis_folder = './'

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer = tf.python_io.TFRecordWriter(os.path.join(dis_folder, 'test.tfrecord'))

data = pd.read_csv(os.path.join(src_folder, 'data.csv'))

for i in range(0, 40):
    img_base = os.path.join(src_folder, 'cam1_I_'+str(i+1))
    print(i)
    if not os.path.isfile(img_base+'_1_color.jpg'):
        print('file not exist!')
        break
    img_1 = np.array(Image.open(img_base+'_1_color.jpg'))
    img_01 = np.array(Image.open(img_base+'_01_color.jpg')).tobytes()
    img_11 = np.array(Image.open(img_base+'_11_color.jpg')).tobytes()
    img_12 = np.array(Image.open(img_base+'_12_color.jpg')).tobytes()
    img_13 = np.array(Image.open(img_base+'_13_color.jpg')).tobytes()
    height, width = img_1.shape[0], img_1.shape[1]
    img_1 = img_1.tobytes()
    dp_file = np.load(img_base+'_1_depth.npy')
    depth_height, depth_width = dp_file.shape[0], dp_file.shape[1]

    example = tf.train.Example(features=tf.train.Features(
        feature={
        'img_01': _bytes_feature(img_01),
        'img_1' : _bytes_feature(img_1),
        'img_11': _bytes_feature(img_11),
        'img_12': _bytes_feature(img_12),
        'img_13': _bytes_feature(img_13),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'move_00': _floats_feature(eval(data['move_00'][i][1:])),
        'move_1': _floats_feature(eval(data['move_1'][i][1:])),
        'rotate_angle': _floats_feature([data['rotate_angle'][i]]),
        'gACT': _int64_feature(data['gACT'][i]),
        'gMOD': _int64_feature(data['gMOD'][i]),
        'gGTO': _int64_feature(data['gGTO'][i]),
        'gSTA': _int64_feature(data['gSTA'][i]),
        'gIMC': _int64_feature(data['gIMC'][i]),
        'gFLT': _int64_feature(data['gFLT'][i]),
        'gPRE': _int64_feature(data['gPRE'][i]),
        'tcp_force': _floats_feature(eval(data['tcp_force'][i][1:])),
        'success': _int64_feature(data['success'][i]),
        'depth_height': _int64_feature(depth_height),
        'depth_width': _int64_feature(depth_width),
        'depth_array_1': _bytes_feature(dp_file.tobytes()),
        'depth_array_01': _bytes_feature(np.load(img_base+'_01_depth.npy').tobytes()),
        'depth_array_11': _bytes_feature(np.load(img_base+'_11_depth.npy').tobytes()),
        'depth_array_12': _bytes_feature(np.load(img_base+'_12_depth.npy').tobytes()),
        'depth_array_13': _bytes_feature(np.load(img_base+'_13_depth.npy').tobytes())
        }))
    writer.write(example.SerializeToString())

writer.close()

index = 0
# init_op = tf.global_variables_initializer()


for serialized_example in tf.python_io.tf_record_iterator("test.tfrecord"):
    print('aaa')
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    m_p_N = example.features.feature['move_00'].float_list.value
    m_i_N = example.features.feature['img_1'].bytes_list.value[0]
    
    img = np.fromstring(m_i_N, dtype=np.uint8).reshape((height, width, -1))
#     # img.size(50,50)
    img = Image.fromarray(img)
    img.save("test"+str(index)+'.jpg')
#     # img.save()
#     #print example
#     print "result from the tfrecord: "
    print m_p_N
    index += 1
    # print m_i_N

# tf_record_filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once('test.tfrecord'))

# tf_record_reader = tf.TFRecordReader()
# _, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

# tf_record_features = tf.parse_single_example(
#     tf_record_serialized,
#     features={
#     'move/position': tf.FixedLenFeature([], tf.float32) 
#     })

# tf_record_position = tf.cast(tf_record_features['move/position'], tf.float32)
# print(tf_record_position)
