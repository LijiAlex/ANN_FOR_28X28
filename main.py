from src.training import training
import tensorflow as tf

mnist = tf.keras.datasets.mnist
training_data, test_data = mnist.load_data()
training(training_data, test_data)
