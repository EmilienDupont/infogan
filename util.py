import keras.backend as K
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

from keras.datasets import mnist
from plotly import tools

EPSILON = 1e-8

def disc_mutual_info_loss(c_disc, aux_dist):
    """
    Mutual Information lower bound loss for discrete distribution.
    """
    reg_disc_dim = aux_dist.get_shape().as_list()[-1]
    cross_ent = - K.mean( K.sum( K.log(aux_dist + EPSILON) * c_disc, axis=1 ) )
    ent = - K.mean( K.sum( K.log(1./reg_disc_dim + EPSILON) * c_disc, axis=1 ) )
    return ent - cross_ent

def get_processed_mnist():
    """
    Get normalized MNIST datasets with correct shapes (Tensorflow style).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))
    return (x_train, y_train), (x_test, y_test)

def plot_digit_grid(model, fig_size=10, digit_size=28, std_dev=2.,
                    filename='infogan'):
    """
    Plot a grid of generated digits. Each column corresponds to a different
    setting of the discrete variable, each row to a random setting of the other
    latent variables.

    Parameters
    ----------
    model : InfoVAE model

    fig_size : int

    digit_size : int

    std_dev : float

    filename : string
    """
    figure = np.zeros((digit_size * fig_size, digit_size * fig_size))
    grid_x = np.linspace(-std_dev, std_dev, fig_size)
    grid_y = np.linspace(-std_dev, std_dev, fig_size)

    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            # Generate a digit and plot it
            generated = model.generate()
            digit = generated[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    trace = go.Heatmap(
                x = grid_x,
                y = grid_y,
                z = figure,
                colorscale='Viridis'
            )

    layout = go.Layout(
        yaxis=dict(
            autorange='reversed'
        )
    )

    fig = go.Figure(data=[trace], layout=layout)

    py.plot(fig, filename=get_timestamp_filename(filename), auto_open=False)

def get_timestamp_filename(filename):
    """
    Returns a string of the form "filename_<date>.html"
    """
    date = time.strftime("%H-%M_%d-%m-%Y")
    return filename + "_" + date + ".html"

def sample_unit_gaussian(num_rows=1, dimension=1):
    return np.random.normal(size=(num_rows, dimension))

def sample_categorical(num_rows=1, num_categories=2):
    sample = np.zeros(shape=(num_rows, num_categories))
    sample[np.arange(num_rows), np.random.randint(num_categories, size=num_rows)] = 1.
    return sample

def get_batch(data, batch_size=1):
    return data[ np.random.randint(data.shape[0], size=batch_size) ]
