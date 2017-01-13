import keras.backend as K
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Convolution2D, Deconvolution2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import objectives
from util import (disc_mutual_info_loss, sample_unit_gaussian,
                  sample_categorical, plot_digit_grid, EPSILON)

class InfoGAN():
    """
    Class to handle building and training InfoGAN models.
    """
    def __init__(self, input_shape=(28, 28, 1), latent_dim=62, reg_cont_dim=0,
                 reg_disc_dim=10, filters=(64, 64, 64),
                 aux_filters=(64, 64, 64), batch_size=128):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_dim : int
            Dimension of latent distribution.

        reg_cont_dim : int
            Dimension of continuous latent regularized distribution.

        reg_disc_dim : int
            Dimension of discrete latent regularized distribution.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of depth.

        aux_filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution for auxiliary distribution
            in increasing order of depth.
        """
        self.generator = None
        self.discriminator = None
        self.batch_size = batch_size
        self.callbacks = []
        self.opt = None
        self.gan = None
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.reg_disc_dim = reg_disc_dim
        self.reg_cont_dim = reg_cont_dim
        self.total_latent_dim = latent_dim + reg_disc_dim + reg_cont_dim
        self._setup_model()

    def _setup_model(self):
        """
        Method to set up model.
        """
        self._setup_generator()
        self._setup_discriminator()
        self._setup_auxiliary()
        self._setup_gan()

    def _setup_generator(self):
        """
        Set up generator G
        """
        self.z_input = Input(batch_shape=(self.batch_size, self.latent_dim), name='z_input')
        self.c_disc_input = Input(batch_shape=(self.batch_size, self.reg_disc_dim), name='c_input')
        x = merge([self.z_input, self.c_disc_input], mode='concat')
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        x = Dense(7 * 7 * 128, activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        x = Reshape((7, 7, 128))(x)
        x = Deconvolution2D(64, 4, 4, output_shape=(self.batch_size, 14, 14, 64),
                            subsample=(2,2), border_mode='same',
                            activation='relu')(x)
        x = BatchNormalization(mode=2)(x)
        self.g_output = Deconvolution2D(1, 4, 4,
                            output_shape=(self.batch_size,) + self.input_shape,
                            subsample=(2,2), border_mode='same',
                            activation='sigmoid', name='generated')(x)

        self.generator = Model(input=[self.z_input, self.c_disc_input], output=self.g_output, name='gen_model')
        self.opt_generator = Adam(lr=1e-3)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt_generator)

    def _setup_discriminator(self):
        """
        Set up discriminator D
        """
        self.d_input = Input(batch_shape=(self.batch_size,) + self.input_shape, name='d_input')
        x = Convolution2D(64, 4, 4, subsample=(2,2),
                          activation=LeakyReLU(0.1))(self.d_input)
        x = Convolution2D(128, 4, 4, subsample=(2,2),
                          activation=LeakyReLU(0.1))(x)
        x = BatchNormalization(mode=2)(x)
        x = Flatten()(x)
        x = Dense(1024, activation=LeakyReLU(0.1))(x)
        self.d_hidden = BatchNormalization(mode=2)(x) # Store this to set up Q
        self.d_output = Dense(1, activation='sigmoid', name='d_output')(self.d_hidden)

        self.discriminator = Model(input=self.d_input, output=self.d_output, name='dis_model')
        self.opt_discriminator = Adam(lr=2e-4)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt_discriminator)

    def _setup_auxiliary(self):
        """
        Setup auxiliary distribution.
        """
        x = Dense(128)(self.d_hidden)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        self.q_output = Dense(self.reg_disc_dim, activation='softmax', name='auxiliary')(x)
        self.auxiliary = Model(input=self.d_input, output=self.q_output, name='aux_model')
        # It does not matter what the loss is here, as we do not specifically train this model
        self.auxiliary.compile(loss='mse', optimizer=self.opt_discriminator)

    def _setup_gan(self):
        """
        Set up GAN
        Discriminator weights should not be trained with the GAN.
        """
        self.discriminator.trainable = False
        gan_output = self.discriminator(self.g_output)
        gan_output_aux = self.auxiliary(self.g_output)
        self.gan = Model(input=[self.z_input, self.c_disc_input], output=[gan_output, gan_output_aux])
        self.gan.compile(loss={'dis_model' : 'binary_crossentropy',
                               'aux_model' : disc_mutual_info_loss},
                         loss_weights={'dis_model' : 1.,
                                       'aux_model' : -1.},
                         optimizer=self.opt_generator)

    def sample_latent_distribution(self):
        """
        Returns continuous and discrete samples from latent distribution
        """
        z = sample_unit_gaussian(self.batch_size, self.latent_dim)
        c_disc = sample_categorical(self.batch_size, self.reg_disc_dim)
        return z, c_disc

    def generate(self):
        """
        Generate a batch of examples.
        """
        z, c_disc = self.sample_latent_distribution()
        return self.generator.predict([z, c_disc], batch_size=self.batch_size)

    def discriminate(self, x_batch):
        """
        """
        return self.discriminator.predict(x_batch, batch_size=self.batch_size)

    def get_aux_dist(self, x_batch):
        """
        """
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size)

    def plot(self):
        """
        """
        return plot_digit_grid(self)
