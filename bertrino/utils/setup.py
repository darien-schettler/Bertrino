from bertrino.utils.general import flatten_l_o_l
from bertrino.configs.data_configs import GCSPathInfo
import tensorflow as tf
import pandas as pd
import json
import os

GCS_PATH_CONFIG = GCSPathInfo()


def load_json(path):
    """Load json file into python dictionary"""
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    """Save python dictionary into json file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def configure_tf_xla():
    print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")

    print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
    # enable XLA optmizations (10% speedup when using @tf.function calls)
    tf.config.optimizer.set_jit(True)

    print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")


def load_sensor_df(sensor_csv_path=None):
    """ Load sensor locations for checking/plotting later. """
    if sensor_csv_path is None:
        sensor_csv_path = os.path.join(
            GCS_PATH_CONFIG.competition_data_path,
            "sensor_geometry.csv"
        )
    return pd.read_csv(sensor_csv_path)


def get_tfrec_paths(n_parts=1):
    """ Retrieve the paths to the tfrecord files for the specified number of parts """
    return flatten_l_o_l([
        tf.io.gfile.glob(os.path.join(GCS_PATH_CONFIG.__dict__[f"tfrecords_part_{n_part}_path"], "*.tfrec")) \
        for n_part in range(1, n_parts+1)
    ])


def get_train_val_split(val_split=0.1, tfrec_paths=None, n_parts=1, approx_events_per_tfrec=70_000, verbose=True):
    """
    Get the paths to the tfrecord files (either the entire dataset or some number of parts (1-6))
    Split the tfrecord files into a train and validation distribution
    Print some information about the train and validation distributions

    Args:
        val_split (float, optional): the fraction of the dataset to use for validation
        tfrec_paths (list of strings, optional): the paths to the tfrecord files
        n_parts (int, optional): the number of parts to use (1-6)
        approx_events_per_tfrec (int, optional): the approximate number of events per tfrecord file
        verbose (bool, optional): whether to print information about the train and validation distributions

    Returns:
        train_tfrec_paths (list of strings): the paths to the train tfrecord files
        val_tfrec_paths (list of strings): the paths to the validation tfrecord files
    """

    # get the paths to the tfrecord files if not provided
    if tfrec_paths is None:
        tfrec_paths = get_tfrec_paths(n_parts=n_parts)

    # get the total number of tfrecord files
    n_total_recs = len(tfrec_paths)

    # get the number of train and validation tfrecord files
    n_val_recs = int(n_total_recs * val_split)
    n_train_recs = n_total_recs - n_val_recs

    # get the paths to the train and validation tfrecord files
    val_tfrec_paths = tfrec_paths[:n_val_recs]
    train_tfrec_paths = tfrec_paths[n_val_recs:]

    # print information about the train and validation distributions if flag set
    if verbose:
        # get the approximate number of events in the train and validation tfrecord files
        approx_val_events = approx_events_per_tfrec * n_val_recs
        approx_train_events = approx_events_per_tfrec * n_train_recs

        print(f"TFRECORDS:\n\tN_TOTAL_RECS --> {n_total_recs}"
              f"\n\tN_VAL_RECS   --> {n_val_recs}  (~{int(val_split * 100)}%)"
              f"\n\tN_TRAIN_RECS --> {n_train_recs}\n")

        print(f"\n... USING {len(train_tfrec_paths)} TRAIN TFRECORDS WITH "
              f"~{approx_events_per_tfrec:,} FILES PER RECORD ...")
        for x in train_tfrec_paths[:5]:
            print(f"\t--> {x.split('/', 2)[-1]}")

        print(f"\n... USING {len(val_tfrec_paths)} VALIDATION TFRECORDS WITH "
              f"~{approx_events_per_tfrec:,} FILES PER RECORD ...")
        for x in val_tfrec_paths[:5]:
            print(f"\t--> {x.split('/', 2)[-1]}")

    return train_tfrec_paths, val_tfrec_paths
