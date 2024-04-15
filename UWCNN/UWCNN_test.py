from UWCNN.model import T_CNN
from UWCNN.utils import *
import numpy as np
import tensorflow as tf
import pprint
import os

from WaterNet.WaterNet_test import flags

def UWCNN_test(filename)->str:
  FLAGS = flags.FLAGS
  image_test = get_image('cache\\'+filename,is_grayscale=False)
  shape = image_test.shape
  with tf.Session() as sess:
    # with tf.device('/cpu:0'):
      srcnn = T_CNN(sess, 
                image_height=shape[0],
                image_width=shape[1],  
                label_height=230, 
                label_width=310, 
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim, 
                checkpoint_dir='UWCNN\\checkpoint',
                sample_dir=FLAGS.sample_dir,
                # test_image_name = test_data_list[ide],
                test_image_name=filename,
                id = 0
                )
      name = srcnn.train(FLAGS)
      sess.close()

  return name

      

