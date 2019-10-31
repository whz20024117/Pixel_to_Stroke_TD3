import tensorflow as tf
from pretrianGAN.config import config
import numpy as np


def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


def discriminator(img_in, rate, reuse=None):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1] + config['IMAGE_DIM'] + [1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x


def generator(z, rate, is_training):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, rate=rate)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x


def deblur(img):
    return np.array([np.array([1 if i > 150 else 0 for i in im]) for im in img]).reshape(-1, 784)


def CNN(X, keep_prob):

    # CONV 1
    W1 = tf.get_variable("classifier/W1", [4, 4, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("classifier/b1", [32], initializer=tf.zeros_initializer())

    Z1 = tf.nn.conv2d(input=X, filter=W1, strides=[1, 1, 1, 1], padding="SAME")
    Z1 = tf.nn.bias_add(Z1, b1)
    A1 = tf.nn.relu(Z1)
    print(A1)

    # MAX POOL 1
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print(P1)

    # CONV 2
    W2 = tf.get_variable("classifier/W2", [2, 2, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("classifier/b2", [64], initializer=tf.zeros_initializer())

    Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=[1, 1, 1, 1], padding="SAME")
    Z2 = tf.nn.bias_add(Z2, b2)
    A2 = tf.nn.relu(Z2)
    print(A2)

    # Flatten
    A2 = tf.contrib.layers.flatten(A2)

    # FULL CONNECT 3
    W3 = tf.get_variable('classifier/W3', [512, A2.shape[1]], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('classifier/b3', [512, 1], initializer=tf.zeros_initializer())
    Z3 = tf.add(tf.matmul(W3, tf.matrix_transpose(A2)), b3)
    A3 = tf.nn.relu(Z3)
    print(A3)

    # FULL CONNECT 4
    W4 = tf.get_variable('classifier/W4', [256, 512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable('classifier/b4', [256, 1], initializer=tf.zeros_initializer())
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    print(Z4)

    # FULL CONNECT 5
    W5 = tf.get_variable('classifier/W5', [17, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable('classifier/b5', [17, 1], initializer=tf.zeros_initializer())
    A4_drop = tf.nn.dropout(A4, keep_prob)
    Z5 = tf.add(tf.matmul(W5, A4_drop), b5)

    Z5 = tf.matrix_transpose(Z5)

    return tf.nn.softmax(Z5)