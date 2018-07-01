import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from random import shuffle

class DCGAN(object):
    def __init__(self, sess,output_height=128, output_width=128,
                             batch_size=64, sample_num = 64,
                             z_dim=100, gf_dim=64, df_dim=64,
                             c_dim=3,
                             checkpoint_dir=None):

        self.sess = sess
        self.output_width = output_width
        self.output_height = output_height
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir

        self.model_setup()

    def model_setup(self):
        # Setup Placeholders
        self.z_placeholder = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z_placeholder')
        self.x_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, self.output_width, self.output_height, self.c_dim],
                                            name='x_placeholder')
        self.y_placeholder = tf.placeholder(tf.float32,
                                            shape=[None, self.output_width, self.output_height, self.c_dim],
                                            name='y_placeholder')

        # Evaluate Discriminator and Generator and Sampler Generator
        self.dx , self.dx_logits = self.dcgan_discriminator(input_images=self.x_placeholder, reuse_variables=False)
        self.ax, self.ax_logits = self.dcgan_abstractor(input_images=self.y_placeholder, reuse_variables=False)

        self.gz = self.dcgan_generator(z=self.z_placeholder, is_train=True, reuse_variables=False)
        self.gz_sampler = self.dcgan_generator(z=self.z_placeholder, is_train=False, reuse_variables=True)

        self.dgz, self.dgz_logits = self.dcgan_discriminator(input_images=self.gz, reuse_variables=True)
        self.agz, self.agz_logits = self.dcgan_abstractor(input_images=self.gz, reuse_variables=True)

        # Define Losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.dx_logits,
                            labels=tf.ones_like(self.dx_logits),
                        name='d_loss_real'))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.dgz_logits,
                            labels=tf.zeros_like(self.dgz_logits),
                        name='d_loss_fake'))

        self.a_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.ax_logits,
                            labels=tf.ones_like(self.ax_logits),
                        name='d_loss_real'))
        self.a_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.agz_logits,
                            labels=tf.zeros_like(self.agz_logits),
                        name='d_loss_fake'))

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.dgz_logits,
                            labels=tf.ones_like(self.dgz_logits),
                        name='g_loss')) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.agz_logits,
                            labels=tf.ones_like(self.agz_logits),
                        name='g_loss'))

        self.d_loss = self.d_loss_fake + self.d_loss_real
        self.a_loss = self.a_loss_fake + self.a_loss_real


        # Trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dcgan_d_' in var.name]
        self.g_vars = [var for var in t_vars if 'dcgan_g_' in var.name]
        self.a_vars = [var for var in t_vars if 'dcgan_a_' in var.name]

        # Saver
        self.saver = tf.train.Saver()


    def dcgan_discriminator(self, input_images, reuse_variables=None):
        with tf.variable_scope('dcgan_d_', reuse=reuse_variables) as scope:
            # DCGAN ARTICLE
            # Use Batch-norm. Avoid apply batch norm to the discriminator first layer
            # Use lReLU act in discriminator for all layers
            # =================================================================================
            # First convolutional layer - input [batch_size, W, H, c_dim] -> output [batch_size, W/2, H/2, df_dim]
            d_w1 = tf.get_variable('d_w1', [5, 5, self.c_dim, self.df_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b1 = tf.get_variable('d_b1',[self.df_dim], initializer=tf.constant_initializer(0.0))
            d1 = tf.nn.conv2d(input=input_images, filter=d_w1, strides = [1,2,2,1], padding='SAME')
            d1 = tf.add(d1,d_b1)
            d1 = tf.nn.leaky_relu(features=d1,alpha=0.2,name='d1_LReLU')

            # Second convolutional layer - input [batch_size, W/2, H/2, df_dim] -> [batch_size, W/4, H/4, df_dim*2]
            d_w2 = tf.get_variable('d_w2', [5, 5, self.df_dim, self.df_dim*2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b2 = tf.get_variable('d_b2', [self.df_dim*2], initializer=tf.constant_initializer(0.0))
            d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
            d2 = tf.nn.bias_add(d2,d_b2)
            d2 = tf.contrib.layers.batch_norm(inputs=d2, decay=0.90, epsilon=1e-5,scope='d2_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d2 = tf.nn.leaky_relu(features=d2, alpha=0.2, name='d2_LReLU')

            # Third convolutional layer - input [batch_size, W/4, H/4, df_dim*2] -> [batch_size, W/8, H/8, df_dim*4]
            d_w3 = tf.get_variable('d_w3', [5, 5, self.df_dim*2, self.df_dim*4],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b3 = tf.get_variable('d_b3', [self.df_dim*4], initializer=tf.constant_initializer(0.0))
            d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 2, 2, 1], padding='SAME')
            d3 = tf.nn.bias_add(d3, d_b3)
            d3 = tf.contrib.layers.batch_norm(inputs=d3, decay=0.90, epsilon=1e-5, scope='d3_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d3 = tf.nn.leaky_relu(features=d3, alpha=0.2, name='d3_LReLU')

            # Four convolutional layer - [batch_size, W/8, H/8, df_dim*4] -> [batch_size, W/16, H/16, df_dim*8]
            d_w4 = tf.get_variable('d_w4', [5, 5, self.df_dim * 4, self.df_dim * 8],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b4 = tf.get_variable('d_b4', [self.df_dim * 8], initializer=tf.constant_initializer(0.0))
            d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 2, 2, 1], padding='SAME')
            d4 = tf.nn.bias_add(d4 , d_b4)
            d4 = tf.contrib.layers.batch_norm(inputs=d4, decay=0.90, epsilon=1e-5, scope='d4_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d4 = tf.nn.leaky_relu(features=d4, alpha=0.2, name='d4_LReLU')

            d_flat = tf.layers.flatten(d4, name='d_flat')

            # Fully Connected Layer
            d_w5 = tf.get_variable('d_w5', [(self.output_height/16)*(self.output_width/16)*self.df_dim*8, 1],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b5 = tf.get_variable('d_b5', [1],
                                   initializer=tf.constant_initializer(0.0))
            d5 = tf.add(tf.matmul(d_flat, d_w5), d_b5)

            return tf.nn.sigmoid(d5, name='d_out_sigmoid'), d5

    def dcgan_abstractor(self, input_images, reuse_variables=None):
        with tf.variable_scope('dcgan_a_', reuse=reuse_variables) as scope:
            # DCGAN ARTICLE
            # Use Batch-norm. Avoid apply batch norm to the discriminator first layer
            # Use lReLU act in discriminator for all layers
            # =================================================================================
            # First convolutional layer - input [batch_size, W, H, c_dim] -> output [batch_size, W/2, H/2, df_dim]
            d_w1 = tf.get_variable('d_w1', [5, 5, self.c_dim, self.df_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b1 = tf.get_variable('d_b1',[self.df_dim], initializer=tf.constant_initializer(0.0))
            d1 = tf.nn.conv2d(input=input_images, filter=d_w1, strides = [1,2,2,1], padding='SAME')
            d1 = tf.add(d1,d_b1)
            d1 = tf.nn.leaky_relu(features=d1,alpha=0.2,name='d1_LReLU')

            # Second convolutional layer - input [batch_size, W/2, H/2, df_dim] -> [batch_size, W/4, H/4, df_dim*2]
            d_w2 = tf.get_variable('d_w2', [5, 5, self.df_dim, self.df_dim*2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b2 = tf.get_variable('d_b2', [self.df_dim*2], initializer=tf.constant_initializer(0.0))
            d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
            d2 = tf.nn.bias_add(d2,d_b2)
            d2 = tf.contrib.layers.batch_norm(inputs=d2, decay=0.90, epsilon=1e-5,scope='d2_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d2 = tf.nn.leaky_relu(features=d2, alpha=0.2, name='d2_LReLU')

            # Third convolutional layer - input [batch_size, W/4, H/4, df_dim*2] -> [batch_size, W/8, H/8, df_dim*4]
            d_w3 = tf.get_variable('d_w3', [5, 5, self.df_dim*2, self.df_dim*4],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b3 = tf.get_variable('d_b3', [self.df_dim*4], initializer=tf.constant_initializer(0.0))
            d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 2, 2, 1], padding='SAME')
            d3 = tf.nn.bias_add(d3, d_b3)
            d3 = tf.contrib.layers.batch_norm(inputs=d3, decay=0.90, epsilon=1e-5, scope='d3_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d3 = tf.nn.leaky_relu(features=d3, alpha=0.2, name='d3_LReLU')

            # Four convolutional layer - [batch_size, W/8, H/8, df_dim*4] -> [batch_size, W/16, H/16, df_dim*8]
            d_w4 = tf.get_variable('d_w4', [5, 5, self.df_dim * 4, self.df_dim * 8],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b4 = tf.get_variable('d_b4', [self.df_dim * 8], initializer=tf.constant_initializer(0.0))
            d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 2, 2, 1], padding='SAME')
            d4 = tf.nn.bias_add(d4 , d_b4)
            d4 = tf.contrib.layers.batch_norm(inputs=d4, decay=0.90, epsilon=1e-5, scope='d4_bn',is_training=True,
                                              scale=True, updates_collections=None)
            d4 = tf.nn.leaky_relu(features=d4, alpha=0.2, name='d4_LReLU')

            d_flat = tf.layers.flatten(d4, name='d_flat')

            # Fully Connected Layer
            d_w5 = tf.get_variable('d_w5', [(self.output_height/16)*(self.output_width/16)*self.df_dim*8, 1],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            d_b5 = tf.get_variable('d_b5', [1],
                                   initializer=tf.constant_initializer(0.0))
            d5 = tf.add(tf.matmul(d_flat, d_w5), d_b5)

            return tf.nn.sigmoid(d5, name='d_out_sigmoid'), d5

    def dcgan_generator(self, z, is_train, reuse_variables):
        with tf.variable_scope('dcgan_g_', reuse=reuse_variables) as scope:
            # DCGAN ARTICLE
            # Use Batch-norm in generator
            # Use ReLU act in generator for all players except for the output, which uses Tanh
            # =================================================================================
            # Fully Connect Layer [batch_size, z_dim] -> [batch_size, H/16, W/16, gf * 8]
            g_w1 = tf.get_variable('g_w1', shape=[self.z_dim, (self.output_height/16)*(self.output_width/16)*8*self.gf_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b1 = tf.get_variable('g_b1', shape=[(self.output_height/16)*(self.output_width/16)*8*self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g1 = tf.add(tf.matmul(z, g_w1), g_b1)

            g1 = tf.reshape(g1, shape=[-1, self.output_width//16, self.output_height//16, 8*self.gf_dim],
                            name='g1_reshape')
            g1 = tf.contrib.layers.batch_norm(inputs=g1, decay=0.90, epsilon=1e-5, scope='g1_bn',is_training=is_train,
                                              scale=True, updates_collections=None)
            g1 = tf.nn.relu(g1,name='g1_ReLu')

            # First Deconv Layer [batch_size, H/16, W/16, gf * 8] -> [batch_size, H/8, W/8, gf * 4]
            g_w2 = tf.get_variable('g_w2', shape=[5, 5, self.gf_dim*4, self.gf_dim*8],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b2 = tf.get_variable('g_b2', shape=[4*self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g2 = tf.nn.conv2d_transpose(g1, filter=g_w2,
                                        output_shape=[self.batch_size, self.output_width//8, self.output_height//8, self.gf_dim*4],
                                        strides=[1,2,2,1])
            g2 = tf.nn.bias_add(g2, g_b2)
            g2 = tf.contrib.layers.batch_norm(inputs=g2, decay=0.90, epsilon=1e-5, scope='g2_bn',is_training=is_train,
                                              scale=True, updates_collections=None)
            g2 = tf.nn.relu(g2, name='g2_ReLu')

            # Second Deconv Layer [batch_size, H/8, W/8, gf * 4] -> [batch_size, H/4, W/4, gf * 2]
            g_w3 = tf.get_variable('g_w3', shape=[5, 5, self.gf_dim * 2, self.gf_dim * 4],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b3 = tf.get_variable('g_b3', shape=[2 * self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g3 = tf.nn.conv2d_transpose(g2, filter=g_w3,
                                        output_shape=[self.batch_size, self.output_width // 4, self.output_height // 4,
                                                      self.gf_dim * 2],
                                        strides=[1, 2, 2, 1])
            g3 = tf.nn.bias_add(g3, g_b3)
            g3 = tf.contrib.layers.batch_norm(inputs=g3, decay=0.90, epsilon=1e-5, scope='g3_bn',is_training=is_train,
                                              scale=True, updates_collections=None)
            g3 = tf.nn.relu(g3, name='g3_ReLu')

            # Third Deconv Layer [batch_size, H/4, W/4, gf * 2] -> [batch_size, H/2, W/2, gf]
            g_w4 = tf.get_variable('g_w4', shape=[5, 5, self.gf_dim, self.gf_dim * 2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b4 = tf.get_variable('g_b4', shape=[self.gf_dim],
                                   initializer=tf.constant_initializer(0.0))
            g4 = tf.nn.conv2d_transpose(g3, filter=g_w4,
                                        output_shape=[self.batch_size, self.output_width // 2, self.output_height // 2,
                                                      self.gf_dim],
                                        strides=[1, 2, 2, 1])
            g4 = tf.nn.bias_add(g4, g_b4)
            g4 = tf.contrib.layers.batch_norm(inputs=g4, decay=0.90, epsilon=1e-5, scope='g4_bn',is_training=is_train,
                                              scale=True, updates_collections=None)
            g4 = tf.nn.relu(g4, name='g4_ReLu')

            # Fourth Deconv Layer [batch_size, H/2, W/2, gf] -> [batch_size, H, W, c_dim]
            g_w5 = tf.get_variable('g_w5', shape=[5, 5, self.c_dim, self.gf_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
            g_b5 = tf.get_variable('g_b5', shape=[self.c_dim],
                                   initializer=tf.constant_initializer(0.0))
            g5 = tf.nn.conv2d_transpose(g4, filter=g_w5,
                                        output_shape=[self.batch_size, self.output_width, self.output_height,
                                                      self.c_dim],
                                        strides=[1, 2, 2, 1])
            g5 = tf.nn.bias_add(g5, g_b5)

            return tf.tanh(g5, 'g_out')


    def train(self, config, data_pathsX, data_pathsY):
        # set optimizers
        d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1)\
                    .minimize(self.d_loss, var_list=self.d_vars)
        a_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1) \
            .minimize(self.a_loss, var_list=self.a_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1)\
                    .minimize(self.g_loss, var_list=self.g_vars)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Load Model if could load
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Create Sample batch
        dataX = data_pathsX
        dataY = data_pathsY

        # Create Sample batch X ===============
        sample_files = dataX[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=config.crop,
                      grayscale=config.grayscale) for sample_file in sample_files]
        if (config.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        x_batch_samples = sample_inputs

        # Create Sample batch Y ================
        sample_files = dataY[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=config.input_height,
                      input_width=config.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=config.crop,
                      grayscale=config.grayscale) for sample_file in sample_files]
        if (config.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        y_batch_samples = sample_inputs

        # Create Sample batch Z ================
        z_batch_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

        # Training==============================
        for epoch in xrange(config.epoch):
            # Shuffle Data
            shuffle(dataX)
            shuffle(dataY)

            # Calculate number of iterations
            batch_idxs_X = min(len(dataX), config.train_size) // config.batch_size
            batch_idxs_Y = min(len(dataY), config.train_size) // config.batch_size # must have the same size


            # For each iteration
            for idx in xrange(0, batch_idxs_X):
                # x_batch
                batch_files = dataX[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=config.input_height,
                              input_width=config.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=config.crop,
                              grayscale=config.grayscale) for batch_file in batch_files]
                if config.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                x_batch = batch_images

                # y_batch
                batch_files = dataY[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(batch_file,
                                input_height=config.input_height,
                                input_width=config.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=config.crop,
                                grayscale=config.grayscale) for batch_file in batch_files]
                if config.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                y_batch = batch_images
                z_batch = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

                # Update D net
                errD, _ = self.sess.run(fetches=[self.d_loss, d_optim],
                                        feed_dict={self.x_placeholder: x_batch, self.z_placeholder: z_batch})
                errA, _ = self.sess.run(fetches=[self.a_loss, a_optim],
                                        feed_dict={self.y_placeholder: y_batch, self.z_placeholder: z_batch})
                # Update 2 times G net
                errG, _ = self.sess.run(fetches=[self.g_loss, g_optim],
                                        feed_dict={self.z_placeholder: z_batch})
                errG, _ = self.sess.run(fetches=[self.g_loss, g_optim],
                                        feed_dict={self.z_placeholder: z_batch})

                # Print some info
                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, a_loss %.8f, g_loss: %.8f" \
                      % (epoch, config.epoch, idx, batch_idxs_X,
                         time.time() - start_time, errD, errA, errG))

                # For each 50 iterations save some samples
                if np.mod(counter, 20) == 1:
                    samples, d_loss, a_loss, g_loss = self.sess.run(
                        [self.gz_sampler, self.d_loss, self.a_loss, self.g_loss],
                        feed_dict={
                            self.z_placeholder: z_batch_samples,
                            self.x_placeholder: x_batch_samples,
                            self.y_placeholder: y_batch_samples,
                        },
                    )

                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                # Save Training
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
                    print("========SAVED==========")

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
          self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
          counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
          print(" [*] Success to read {}".format(ckpt_name))
          return True, counter
        else:
          print(" [*] Failed to find a checkpoint")
          return False, 0

    def get_samples(self, sample_dir, option=1):
        if option == 1:
            image_frame_dim = int(math.ceil(self.batch_size ** .5))
            z_batch_samples = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
            samples = self.sess.run(
                [self.gz_sampler],
                feed_dict={
                    self.z_placeholder: z_batch_samples,
                },
            )
            print(np.squeeze(np.array(samples), axis=0).shape)
            save_images(np.squeeze(np.array(samples), axis=0), [image_frame_dim,image_frame_dim],
                        './{}/inference_{}.png'.format(sample_dir, time.time()))
        elif option == 2:
                z_batch = np.random.normal(loc=0.0, scale=1.0,
                                                   size=(self.batch_size, self.z_dim)).astype(np.float32)
                image_samples = self.sess.run(fetches=[self.gz_sampler],
                                              feed_dict={self.z_placeholder: z_batch})
                samples = np.squeeze(np.array(image_samples), axis=0)
                for i in xrange(self.batch_size):
                    scipy.misc.imsave('./{}/inference_{}.png'.format(sample_dir, time.time()), inverse_transform(samples[i,:,:,:]))
