import tensorflow as tf
import numpy as np
import random
import os


def flatten_l_o_l(nested_list):
    """Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
