import tensorflow.keras as keras
from tensorflow.keras import layers, models
from Config import *

z_dim = model_config.z_dim
num_classes = model_config.num_classes
image_shape = model_config.imag_shape


class Generator(models.Model):
    def __init__(self, z_dim=100, num_classes=10):
        label = layers.Input((1,))
        li = layers.Embedding(num_classes, 50)(label)
        li = layers.Dense(7 * 7)(li)
        li = layers.Reshape((7, 7, 1))(li)

        latent = layers.Input((z_dim,))
        k = layers.Dense(7 * 7 * 128)(latent)
        k = layers.LeakyReLU()(k)
        k = layers.Reshape((7, 7, 128))(k)

        model_input = layers.Concatenate()([li, k])  # label, latent 순으로 병합 7,7,33

        h = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(model_input)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        gen_output = layers.Conv2DTranspose(1, (7, 7), strides=1, padding='same', activation='tanh')(h)

        super().__init__([label, latent], gen_output)  # label, latent 순으로 인풋


class Discriminator(models.Model):
    def __init__(self, z_dim=100, imag_shape=(28, 28, 1)):
        label = layers.Input((1,))
        li = layers.Embedding(num_classes, 50)(label)
        li = layers.Dense(28 * 28)(li)
        li = layers.Reshape((28, 28, 1))(li)

        imag = layers.Input(imag_shape)

        model_input = layers.Concatenate()([li, imag])  # 28,28,2

        h = layers.Conv2D(128, (3, 3), strides=2, padding='same')(model_input)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2D(128, (3, 3), strides=2, padding='same')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Flatten()(h)
        h = layers.Dropout(0.3)(h)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__([label, imag], y)
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))


class CGAN(models.Model):
    def __init__(self):
        generator = Generator()
        discriminator = Discriminator()

        cgan_label_input = layers.Input((1,))
        cgan_latent_input = layers.Input((z_dim))

        cgan_output = discriminator([cgan_label_input, generator([cgan_label_input, cgan_latent_input])])

        super().__init__([cgan_label_input, cgan_latent_input], cgan_output)
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

        self.generator = generator
        self.discriminator = discriminator
