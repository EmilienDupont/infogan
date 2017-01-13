import numpy as np
from util import get_batch

class Trainer():
    def __init__(self, model):
        self.model = model
        self.d_loss_history = []
        self.g_loss_history = []
        self.ent_loss_history = []

    def train_gan(self):
        z, c_disc = self.model.sample_latent_distribution()
        # Generate examples that fool classifier, i.e. where D outputs 1
        target = np.ones(self.model.batch_size).astype(int)
        gan_loss = self.model.gan.train_on_batch([z, c_disc], [target, c_disc])
        self.g_loss_history.append(gan_loss[1])
        self.ent_loss_history.append(gan_loss[2])

    def train_discriminator(self, x_batch=None):
        # Train discriminator on fake data
        if x_batch is None:
            fake_batch = self.model.generate()
            # Fake examples, so D should output 0
            target = np.zeros(self.model.batch_size).astype(int)
            d_loss = self.model.discriminator.train_on_batch(fake_batch, target)
        # Or train discriminator on real data
        else:
            # Real data, so D should output 1
            target = np.ones(self.model.batch_size).astype(int)
            d_loss = self.model.discriminator.train_on_batch(x_batch, target)
        self.d_loss_history.append(d_loss)

    def fit(self, x_train, num_epochs=1, print_every=0):
        """
        Method to train GAN.

        Parameters
        ----------
        print_every : int
            Print loss information every |print_every| number of batches. If 0
            prints nothing.
        """
        num_batches = x_train.shape[0] / self.model.batch_size
        print("num batches {}".format(num_batches))

        for epoch in range(num_epochs):
            print("\nEpoch {}".format(epoch + 1))

            for batch in range(num_batches):
                x_batch = get_batch(x_train, self.model.batch_size)
                self.train_discriminator()
                self.train_discriminator(x_batch)
                self.train_gan()
                if print_every and batch % print_every == 0:
                    print("GAN loss {} \t D loss {} \t Entropy {}".format(self.g_loss_history[-1], self.d_loss_history[-1], self.ent_loss_history[-1]))
