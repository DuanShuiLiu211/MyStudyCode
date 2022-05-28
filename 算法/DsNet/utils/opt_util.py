from keras.optimizers import RMSprop,SGD,Adam


def generate_optimizer(config):
    opt_name = config['optimizer']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    decay = config['decay']

    if opt_name == 'rms':
        return RMSprop(lr=learning_rate,decay=decay)
    elif opt_name == 'sgd':
        return SGD(lr=learning_rate,momentum=momentum,decay=decay)
    elif opt_name == 'adam':
        return Adam(lr=learning_rate,decay=decay)
    else:
        return None