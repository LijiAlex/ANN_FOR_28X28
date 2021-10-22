def get_data(training_data, test_data, validation_datasize, logger):
    """
    Normalizes the pixel value to lie between 0-1
    :param training_data: training data which contains both input and expected output
    :param test_data: test data which contains both input and expected output
    :param validation_datasize: data to be used for validation
    :param logger: logging object
    :return: training data, validation data and test data
    """
    (X_train_full, y_train_full) = training_data[0], training_data[1]
    (X_test, y_test) = test_data[0], test_data[1]
    logger.info("Data loaded")
    # normalizing and reducing the pixel value to be between 0-1
    X_val, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_val, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]
    X_test = X_test / 255.
    logger.info("Create validation data, training data and test data")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
