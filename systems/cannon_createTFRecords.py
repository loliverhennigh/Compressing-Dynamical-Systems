



import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

import numpy as np 
import tensorflow as tf 
import matplotlib.animation as animation 
from glob import glob as glb

import cannon as cn

FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_converted_frame(cap, shape):
  ret, frame = cap.read()
  frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  return frame

def generate_tfrecords(num_samples, seq_length):

  shape = (28,28)
  frame_num = 4

  filename = '../data/tfrecords/cannon/cannon_num_samples_' + str(num_samples) + '_seq_length_' + str(seq_length) + '.tfrecords'

  tfrecord_filename = glb('../data/tfrecords/cannon/*')  
  if filename in tfrecord_filename:
    print('already a tfrecord there! I will skip this one')
    return

  writer = tf.python_io.TFRecordWriter(filename)

  k = cn.Cannon()
  print(seq_length)
  seq_frames = np.zeros((seq_length, shape[0], shape[1], frame_num))

  ind = 0 
  print('now generating tfrecords ' + filename)
 
  for i in xrange(num_samples):
    seq_frames = k.generate_28x28x4(seq_length, frame_num)
    seq_frames = np.uint8(seq_frames)
    seq_frames_flat = seq_frames.reshape([1,seq_length*28*28*4])
    seq_frame_raw = seq_frames_flat.tostring()
    # create example and write it
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(seq_frame_raw)})) 
    writer.write(example.SerializeToString()) 

    # print status
    ind = ind + 1
    if ind%10000 == 0:
      print('percent converted = ', str(100.0 * float(ind) / num_samples))



