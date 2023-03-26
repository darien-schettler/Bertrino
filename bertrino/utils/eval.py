import numpy as np
import tensorflow as tf

def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """
    Calculate the mean absolute error (MAE) of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors, and then their scalar
    product is computed, which is equal to the cosine of the angle between the two vectors.
    The inverse cosine (arccos) thereof is then the angle between the two input vectors.

    Parameters:
    -----------
    az_true : float or array-like
        True azimuth value(s) in radian
    zen_true : float or array-like
        True zenith value(s) in radian
    az_pred : float or array-like
        Predicted azimuth value(s) in radian
    zen_pred : float or array-like
        Predicted zenith value(s) in radian

    Returns:
    --------
    dist : float
        Mean over the angular distance(s) in radian
    """
    # Ensure all inputs are finite
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")

    # Pre-compute all sine and cosine values
    sa1, ca1 = np.sin(az_true), np.cos(az_true)
    sz1, cz1 = np.sin(zen_true), np.cos(zen_true)
    sa2, ca2 = np.sin(az_pred), np.cos(az_pred)
    sz2, cz2 = np.sin(zen_pred), np.cos(zen_pred)

    # Compute scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + cz1 * cz2

    # Clip scalar product between -1 and 1 to account for numerical instability
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # Convert scalar product back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


def tf_angular_dist_score(labels, preds):
    """
    Calculate the mean absolute error (MAE) of the angular distance between two directions in TensorFlow.
    The two vectors are first converted to cartesian unit vectors, and then their scalar
    product is computed, which is equal to the cosine of the angle between the two vectors.
    The inverse cosine (arccos) thereof is then the angle between the two input vectors.

    Parameters:
    -----------
    labels : tf.Tensor
        A tensor containing the true azimuth and zenith values in radian
    preds : tf.Tensor
        A tensor containing the predicted azimuth and zenith values in radian

    Returns:
    --------
    dist : tf.Tensor
        Mean over the angular distance(s) in radian
    """
    # Break tuple into component vectors
    az_true, zen_true = labels[..., 0], labels[..., 1]
    az_pred, zen_pred = preds[..., 0], preds[..., 1]

    # Pre-compute all sine and cosine values
    sa1, ca1 = tf.math.sin(az_true), tf.math.cos(az_true)
    sz1, cz1 = tf.math.sin(zen_true), tf.math.cos(zen_true)
    sa2, ca2 = tf.math.sin(az_pred), tf.math.cos(az_pred)
    sz2, cz2 = tf.math.sin(zen_pred), tf.math.cos(zen_pred)

    # Compute scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + cz1 * cz2

    # Clip scalar product between -1 and 1 to account for numerical instability
    scalar_prod = tf.clip_by_value(scalar_prod, -1, 1)

    # Convert scalar product back to an angle (in radian)
    return tf.math.abs(tf.math.acos(scalar_prod))