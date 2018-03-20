from utils import read_data, make_label, make_topg
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import sklearn.metrics

xrange = range


class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 ci_dim=2,
                 co_dim=1,
                 scale_factor=None,
                 checkpoint_dir=None,
                 sample_dir=None,
                 model_carac=None,
                 train_dir=None,
                 test_dir=None):

        self.scale_factor=scale_factor
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.model_carac=model_carac
        self.train_dir=train_dir
        self.test_dir=test_dir

        self.ci_dim =ci_dim
        self.co_dim=co_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):


        self.weights = {}
        self.biases = {}
        self.images= tf.placeholder(tf.float32, [None, self.image_size,
                                                                    self.image_size, self.ci_dim], name='images')
        self.labels=tf.placeholder(tf.float32, [None, self.label_size,
                                                                  self.label_size, self.co_dim], name='labels')

        for k in range(self.scale_factor // 2):
            self.weights['w1-%i'%k]=tf.Variable(tf.random_normal([9, 9, self.ci_dim, 64], stddev=1e-3), name='w1-%i'%k)
            self.weights['w2-%i'%k]=tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2-%i'%k)
            self.weights['w3-%i'%k]=tf.Variable(tf.random_normal([5, 5, 32, self.co_dim], stddev=1e-3), name='w3-%i'%k)

            self.biases['b1-%i'%k]=tf.Variable(tf.zeros([64]), name='b1-%i'%k)
            self.biases['b2-%i'%k]=tf.Variable(tf.zeros([32]), name='b2-%i'%k)
            self.biases['b3-%i'%k]=tf.Variable(tf.zeros([1]), name='b3-%i'%k)

        # Loss function (MSE)
        #self.loss = tf.reduce_mean(tf.abs(self.labels - self.pred))  #l1 loss
         #l2

        self.saver = tf.train.Saver()

    def loss(self, step):
        return tf.reduce_mean(tf.square(self.labels - self.model(step)))

    def model(self, step):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1-%i'%step], strides=[1, 1, 1, 1], padding='SAME')
            + self.biases['b1-%i'%step])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2-%i'%step], strides=[1, 1, 1, 1], padding='SAME')
            + self.biases['b2-%i'%step])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3-%i'%step], strides=[1, 1, 1, 1], padding='SAME') \
                + self.biases['b3-%i'%step]
        return conv3

    # def train_op(self, step, config):
    #     return tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss(step))

    def train(self, config):

        if config.is_train:
            data_dir =  self.train_dir
        else:
            data_dir = self.test_dir

        data, label = read_data(data_dir)


        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   #1000, 0.001, staircase=True)
        train_op={}
        models={}
        for step in range(config.scale_factor//2):
            train_op['train_op%i'%step]=tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss(step))
            models['model%i'%step]=self.model(step)



        tf.global_variables_initializer().run()


        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            topg=[make_topg(data, step, self.scale_factor) for step in range(self.scale_factor//2)]
            topg=np.array(topg)
            label=[make_label(label, step, self.scale_factor) for step in range(self.scale_factor//2)]
            label=np.array(label)

            for ep in xrange(config.epoch):
                ep=np.int64(ep)
                # Run by batch images
                batch_idxs = len(data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_images0 = data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels0 = label[:,idx * config.batch_size: (idx + 1) * config.batch_size,:,:]
                    topg0=topg[:,idx * config.batch_size: (idx + 1) * config.batch_size,:,:]

                    counter += 1

                    for step in range(self.scale_factor//2):
                        if step==0:
                            batch_images=np.stack([batch_images0[:,:,:,0],topg0[step]]
                                                   ,axis=-1)
                            batch_labels=batch_labels0[step]
                            _, result=self.sess.run([train_op['train_op%i'%step], models['model%i'%step]],
                                                   feed_dict={self.images: batch_images, self.labels: batch_labels})

                        else:
                            #result = self.sess.run(self.model(step-1),
                                                                 #feed_dict={self.images['images%i'%(step-1)] : batch_images})
                            result=result[:,:,:,0]
                            batch_images=np.stack([result, topg0[step]], axis=-1)
                            batch_labels =batch_labels0[step]
                            _, result=self.sess.run([train_op['train_op%i'%step], models['model%i'%step]],
                                          feed_dict={self.images: batch_images,
                                                     self.labels: batch_labels})



                    if counter % 10 == 0:
                        s = (self.scale_factor // 2) - 1
                        result = self.sess.run(models['model%i'%s], feed_dict={self.images: batch_images})
                        MSE = 0
                        for k in range(config.batch_size):
                            acc = sklearn.metrics.mean_squared_error(result[k, :, :, 0], label[-1,k, :, :, 0])
                            MSE += acc
                        err = MSE / len(data[:, :, :, 0]-1)
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]  " \
                              % ((ep + 1), counter, time.time() - start_time, err))


                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
                tf.get_default_graph().finalize()


        else:
            print("Testing...")

            MSE=0
            for k in range(len(data[:,0,0,0])):
                acc = sklearn.metrics.mean_squared_error(data[k,:,:,0],label[k,:,:,0])
                MSE+=acc
            print("Bicubic error : ", MSE / len(data[:, :, :, 0]))

            for step in range(self.scale_factor//2):
                if step == 0:
                    batch_images = np.stack([data[:, :, :, 0], make_topg(data, step, self.scale_factor)]
                        , axis=-1)

                else :
                    batch_images=np.stack([result, make_topg(data,step,self.scale_factor)], axis=-1)
                result = self.sess.run(models['model%i'%step], feed_dict={self.images: batch_images})
                result = np.array(result)[:,:,:,0]


            MSE=0
            for k in range(len(data[:,0,0,0])):
                acc = sklearn.metrics.mean_squared_error(result[k,:,:],label[k,:,:,0])
                MSE+=acc
            print("Model error : ", MSE/len(data[:,:,:,0]))
            if config.save_result:
                if not os.path.exists(config.result_fold):
                    os.makedirs(config.result_fold)
                for k in range(len(data[:,0,0,0])):
                    original = plt.contourf(label[k, :, :, 0])
                    plt.colorbar(original)
                    Cmap=original.get_cmap()
                    plt.savefig('%s/%d_label.png' % (config.result_fold, k))
                    plt.show()
                    base = plt.contourf(data[k, :, :, 0], cmap=Cmap)
                    plt.colorbar(original)
                    plt.savefig('%s/%d_bicubic.png' %(config.result_fold,k))
                    plt.show()
                    prev=plt.contourf(result[k,:,:], cmap=Cmap)
                    plt.colorbar(original)
                    plt.savefig('%s/%d_predicted.png' %(config.result_fold,k))
                    plt.show()




    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.model_carac)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.model_carac)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False