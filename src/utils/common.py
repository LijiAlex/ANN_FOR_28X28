import yaml
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def get_unique_filename(filename, path_dir):
    """
    Returns an unique filename based on timestamp.
    :param filename: added as suffix to the unique filename.
    :param path_dir: path to the directory where file has to be present
    :return path: path to the unique filename generated
    """
    unique_filename = time.strftime(f"%Y-%m-%d_%H%S%M_{filename}")
    path = os.path.join(path_dir, unique_filename)
    return path


def read_config(config_path):
    """
    Reads the yaml config file.
    :param config_path: Path to the config yaml file.
    :return: yaml file content
    """
    with open(config_path, 'r') as config_file:
        content = yaml.safe_load(config_file)
    return content


def save_plot(history, plot_name, plot_dir, logger):
    """
    saves the plot on the trained data model.
    :param history: history of model training
    :param plot_name: name of the plot
    :param plot_dir: location where plot has to be saved
    :param logger: logging object
    :return: None
    """
    path_to_model = get_unique_filename(plot_name, plot_dir)
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.grid(True)
    plt.savefig(path_to_model)
    logger.info("Plot saved")
