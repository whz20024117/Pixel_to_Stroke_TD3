from pretrianGAN.DataLoader import DataLoader
from pretrianGAN.utils import *
import tensorflow as tf
import numpy as np
from pretrianGAN.config import config


def main(sess):
    dloader = DataLoader(config['DATA_PATH'])

    tf.reset_default_graph()
    batch_size = config['BATCH_SIZE']
    n_noise = config['N_NOISE']

    X_in = tf.placeholder(dtype=tf.float32, shape=[None] + config['IMAGE_DIM'], name='X')
    noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

    rate = tf.placeholder(dtype=tf.float32, name='rate')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    g = generator(noise, rate=rate, is_training=is_training)
    d_real = discriminator(X_in, rate=rate)
    d_fake = discriminator(g, rate,reuse=True)

    vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
    g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

    loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
    loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)

    loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))
    loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d)
        optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g)

    sess.run(tf.global_variables_initializer())

    for i in range(60000):
        train_d = True
        train_g = True
        rate_train = 0.4

        n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
        batch = [np.reshape(b, config['IMAGE_DIM']) for b in dloader.next_batch(batch_size)]

        d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d],
                                                    feed_dict={X_in: batch, noise: n, rate: rate_train,
                                                               is_training: True})

        d_real_ls = np.mean(d_real_ls)
        d_fake_ls = np.mean(d_fake_ls)
        g_ls = g_ls
        d_ls = d_ls

        if g_ls * 1.5 < d_ls:
            train_g = False
            pass
        if d_ls * 2 < g_ls:
            train_d = False
            pass

        if train_d:
            sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, rate: rate_train, is_training: True})

        if train_g:
            sess.run(optimizer_g, feed_dict={noise: n, rate: rate_train, is_training: True})

        if i % 500 == 0:
            print("Generator loss: ", g_ls, "Discriminator loss: ", d_ls, "Step: ", i)


if __name__ == '__main__':
    sess = tf.Session()
    main(sess)
    saver = tf.train.Saver()
    saver.save(sess, config['SAVE_PATH'])
