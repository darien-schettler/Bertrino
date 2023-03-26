import tensorflow as tf


def decode_tfrecords(serialized_example, is_test=False, max_seq_len=128):
    """ Parses a set of features and label from the given serialized_example.

    Args:
        serialized_example (tf.Example): A serialized example containing the
            following features:
                – event_id
                – sensor_ids
                – p_charges
                – p_times
                – p_auxs
                – azimuth (optional, only for non-test sets)
                – zenith (optional, only for non-test sets)
        is_test (bool, optional): Whether to exclude label features (azimuth and zenith)
        max_seq_len (int, optional): The maximum sequence length for sensor_ids, p_charges, p_times, and p_auxs

    Returns:
        tuple: A decoded tf.data.Dataset object representing the tfrecord dataset. The tuple structure depends on
               whether `is_test` is True or False.
    """
    # Define a dictionary of features with their corresponding data types and default values
    feature_dict = {
        'event_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        'sensor_ids': tf.io.FixedLenFeature(shape=[max_seq_len], dtype=tf.int64, default_value=[0] * max_seq_len),
        'p_charges': tf.io.FixedLenFeature(shape=[max_seq_len], dtype=tf.float32, default_value=[0.0] * max_seq_len),
        'p_times': tf.io.FixedLenFeature(shape=[max_seq_len], dtype=tf.float32, default_value=[0.0] * max_seq_len),
        'p_auxs': tf.io.FixedLenFeature(shape=[max_seq_len], dtype=tf.int64, default_value=[0] * max_seq_len),
    }

    if not is_test:
        feature_dict['azimuth'] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0)
        feature_dict['zenith'] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0)

    # Parse the serialized example using the defined feature dictionary
    features = tf.io.parse_single_example(serialized_example, features=feature_dict)

    inputs = (features["sensor_ids"], features["p_times"], features["p_auxs"], features["p_charges"])
    if not is_test:
        # inputs, labels, event_ids
        return inputs, (features["azimuth"], features["zenith"]), (features["event_id"])
    else:
        # inputs, event_ids
        return inputs, (features["event_id"])


def get_tfdataset(tfrec_paths, config, is_test=False, use_autotune=True, return_event_ids=False):
    """ Returns a tf.data.Dataset object representing the tfrecord dataset ready for MLM training.

    Args:
        tfrec_paths (list): A list of paths to the tfrecord files.
        config (BertrinoConfig): A BertrinoConfig object containing the configuration parameters.
        is_test (bool, optional): Whether to exclude label features (azimuth and zenith)
        use_autotune (bool, optional): Whether to use tf.data.AUTOTUNE for parallelization.
        return_event_ids (bool, optional): Whether to return the event_ids as part of the dataset.

    Returns:
        tf.data.Dataset: A tf.data.Dataset object representing the tfrecord dataset ready for MLM training.
    """
    _AT = tf.data.AUTOTUNE if use_autotune else None
    ds = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=_AT)
    ds = ds.map(lambda x: (decode_tfrecords(x, is_test)), num_parallel_calls=_AT) \
           .shuffle(config.n_shuffle_buffer) \
           .batch(config.batch_size, drop_remainder=True) \
           .map(lambda x,y,e: (mask_inputs_tfdata(x,y,e,mg_layer,do_masking=config.DO_MLM_TRAINING, use_aux=USE_AUX))) \
           .prefetch(_AT)
    if return_event_ids is False:
        ds = ds.map(lambda x,y,w,e: (x,y,w), num_parallel_calls=_AT)
    return ds


def print_check_tfdataset(tf_ds):
    """ Prints the first batch of a tf.data.Dataset object. """
    print("\n\n... CHECK FIRST BATCH ...\n")
    # if we have event ids
    if len(tf_ds.take(1)) is 4:
        for x,y,w,e in tf_ds.take(1):
            print("MASKED INPUT TOKENS   : ", x[0, :14].numpy())
            print("ORIGINAL INPUT TOKENS : ", y[0][0, :14].numpy())
            print("MASK WEIGHTINGS.      : ", w[0, :14].numpy())
            print("EVENT IDS             : ", e[:14].numpy())
            break
    else:
        for x,y,w in tf_ds.take(1):
            print("MASKED INPUT TOKENS   : ", x[0, :14].numpy())
            print("ORIGINAL INPUT TOKENS : ", y[0, :14].numpy())
            print("MASK WEIGHTINGS.      : ", w[0, :14].numpy())
            break