## Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 #######
## Project: https://li-chongyi.github.io/proj_benchmark.html 
############################################################################################################################################################################

from WaterNet.model import T_CNN
from WaterNet.utils import *
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 120, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 112, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 112, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 112, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 112, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")

def WaterNet_test(name)->str:
  FLAGS = flags.FLAGS
  #文件路径
  test_image_path = 'cache\\'+name
  image_test =  get_image(test_image_path,is_grayscale=False)
  shape = image_test.shape
  tf.reset_default_graph()
  with tf.Session() as sess:
    # with tf.device('/cpu:0'):
      srcnn = T_CNN(sess, 
                image_height=shape[0],
                image_width=shape[1],  
                label_height=FLAGS.label_height, 
                label_width=FLAGS.label_width, 
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim, 
                c_depth_dim=FLAGS.c_depth_dim,
                checkpoint_dir='WaterNet\\checkpoint',
                sample_dir=FLAGS.sample_dir,
                test_image_name = test_image_path,
                name = name,
                id = 0
                )

      output_name = srcnn.train(FLAGS)
      sess.close()
  tf.get_default_graph().finalize()
  return output_name

