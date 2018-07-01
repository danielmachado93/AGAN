import os
import scipy.misc
import numpy as np
from glob import glob

from model import DCGAN
from utils import pp, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", 9000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 36, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 128, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 128, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 128, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 128, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "The number of input channels")
flags.DEFINE_boolean("grayscale", False, "is gray")
flags.DEFINE_string("datasetX", "datasetX", "The name of dataset for discriminator")
flags.DEFINE_string("datasetY", "datasetY", "The name of dataset for abstractor")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main():
    # Define Data for training
    dataX = glob(os.path.join("./data", FLAGS.datasetX, FLAGS.input_fname_pattern))
    dataY = glob(os.path.join("./data", FLAGS.datasetY, FLAGS.input_fname_pattern))

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        dcgan = DCGAN(
            sess,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            z_dim=FLAGS.generate_test_images,
            c_dim=FLAGS.c_dim,
            checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.train:
            dcgan.train(FLAGS, dataX, dataY)
        else: # INFERENCE
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            # Render samples to "samples" folder
            # Option 1 render manifold of samples with dim = n*n = number_of_samples
            # Option 2 render imagens one by one
            dcgan.get_samples(sample_dir = FLAGS.sample_dir, option=1)

        print("====DONE=====")
main()