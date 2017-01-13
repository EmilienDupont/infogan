from util import get_processed_mnist
from infogan import InfoGAN
from trainer import Trainer

(x_train, y_train), (x_test, y_test) = get_processed_mnist()
model = InfoGAN()
model.gan.summary()
model.discriminator.summary()
trainer = Trainer(model)
trainer.fit(x_train, print_every=10)