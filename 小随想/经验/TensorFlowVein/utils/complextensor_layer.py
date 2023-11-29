import tensorflow as tf


def max_pooling(feature_map, size, stride):
    """max_pooling"""
    mode = feature_map.ndims
    print("tensorshape is %d" % mode)
    if mode >= 2 and feature_map.shape[0] >= stride and feature_map.shape[1] >= stride:
        height = feature_map.shape[0]
        width = feature_map.shape[1]
        out_height = tf.constant([(height - size) // stride + 1], tf.float32)
        out_width = tf.constant([(width - size) // stride + 1], tf.float32)
        out_pooling = tf.zeros((out_height, out_width), dtype=tf.float32)

        i = j = 0
        for m in range(0, height, stride):
            for n in range(0, width, stride):
                if (n + stride) <= width and (m + stride) <= height:
                    out_pooling[i][j] = tf.reduce_max(
                        tf.abs(feature_map[m : m + size, n : n + size])
                    )
                    j += stride
            i += stride
            j = 0

    else:
        width = feature_map.shape[1]
        out_width = tf.constant([(width - size) // stride + 1], tf.float32)
        out_pooling = tf.zeros((1, out_width), dtype=tf.float32)

        k = 0
        for idx in range(0, width, stride):
            if (idx + stride) <= width:
                out_pooling[k] = tf.reduce_max(tf.abs(feature_map[idx : idx + size]))
                k += stride

    return out_pooling
