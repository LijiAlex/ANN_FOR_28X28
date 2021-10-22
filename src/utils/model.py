import tensorflow as tf
from src.utils.common import get_unique_filename


def create_model(loss_function, optimizer, metrics, logger):
    """
    Creates a sequential ANN model with input layer which expects np array of shape 28X28,
    300 hidden layer with relu activation function,
    100 hidden layer with relu activation function and
    10 output layer with softmax function
    :param loss_function: loss function to be used
    :param optimizer: optimizer to be used
    :param metrics: metrics to be used
    :param logger: logging object
    :return:
    """
    layers = [
          tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(layers)
    model_clf.summary()
    model_clf.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    logger.info("Model created")
    return model_clf  # untrained model


def save_model(model, model_name, model_dir, logger):
    path_to_model = get_unique_filename(model_name, path_dir=model_dir)
    model.save(path_to_model)
    logger.info("Model saved")
