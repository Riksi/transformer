import tensorflow as tf


class SequenceMask:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        # Disregards padded elements
        # x: (B, N)
        if isinstance(self.pad, int):
            mask = tf.not_equal(x, self.pad)
        else:
            mask = tf.reduce_all(tf.not_equal(x[..., None], self.pad), axis=-1)
        # Same mask for every position
        # (B, 1, 1, N)
        return mask[:, None, None]


class TargetMask(SequenceMask):
    def __call__(self, x):
        # Disregards "future" elements and any others
        # which are padded
        # (B, N)
        # (B, 1, N)
        pad_mask = super().__call__(x)
        seq_length = tf.shape(x)[-1]
        # Mask shared for same position across batches
        # (N, N)
        # lower_triangular matrix
        future_mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        future_mask = tf.cast(future_mask, tf.bool)
        # (B, 1, 1, N) & (N, N) -> (B, 1, N, N)
        mask = tf.logical_and(pad_mask, future_mask)
        return mask