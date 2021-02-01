# batched data
import tensorflow as tf
import numpy as np
from model import *
from get_data import *
import matplotlib.pyplot as plt

discriminator = Discriminator()

cgan = CGAN()

@tf.function
def train_step(x, y):
    batch_size = x.shape[0]
    with tf.GradientTape() as dis_tp, tf.GradientTape() as gen_tp:
        latent = np.random.normal(0, 1., [batch_size, z_dim])
        fake_images = cgan.generator([y, latent], training=True)
        real_outputs = cgan.discriminator([y, x], training=True)
        fake_outputs = cgan.discriminator([y, fake_images], training=True)

        dis_loss = 0.5 * (keras.losses.binary_crossentropy(tf.ones_like(real_outputs),
                                                           real_outputs) + keras.losses.binary_crossentropy(
            tf.zeros_like(fake_outputs), fake_outputs))

        gan_fake_outputs = cgan([y, latent], training=True)

        gen_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_outputs), gan_fake_outputs)

    cgan.discriminator.trainable = True
    dis_grad = dis_tp.gradient(dis_loss, cgan.discriminator.trainable_variables)

    cgan.discriminator.optimizer.apply_gradients(zip(dis_grad, cgan.discriminator.trainable_variables))

    cgan.discriminator.trainable = False
    gen_grad = gen_tp.gradient(gen_loss, cgan.trainable_variables)
    cgan.optimizer.apply_gradients(zip(gen_grad, cgan.trainable_variables))

    return dis_loss, gen_loss


def generate_and_save_images(model, epoch):
    # `training`이 False : (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됨
    seed = np.random.normal(0, 1., [16, noise_dim])
    cate = train_labels[:16]
    predictions = model([cate, seed], training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray_r')
        plt.axis('off')