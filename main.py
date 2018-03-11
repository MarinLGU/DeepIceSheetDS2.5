from model import SRCNN


import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags

###Model Parameters
flags.DEFINE_integer("epoch", 20000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 41, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 41, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("ci_dim", 2, "Number of channels of the input")
flags.DEFINE_integer("co_dim",1, "Number of channels of the output")
###Save folder parameters
flags.DEFINE_string("train_dir","train_x4","Train h5 file directory")
flags.DEFINE_string("test_dir","test_x4","Train h5 file directory")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("model_carac","x4_withtopg_Adam_20000_e4","Define model")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("result_fold","x4_results_20000_adam_e4","save folder for image is 'save_result'")
###Booleans
flags.DEFINE_boolean("is_train",False, "True for training, False for testing [True]")
flags.DEFINE_boolean("save_result", False, "save predicted and label image")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      ci_dim=FLAGS.ci_dim,
                      co_dim=FLAGS.co_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      model_carac=FLAGS.model_carac,
                      train_dir=FLAGS.train_dir,
                      test_dir=FLAGS.test_dir)

        srcnn.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()