import tensorflow as tf


def smooth_labels(labels, eps=0.1):
    num_classes = tf.cast(tf.shape(labels)[-1], labels.dtype)
    labels = labels * (1 - eps) + eps / num_classes
    return labels


class MaskedLoss(object):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, y_true, y_pred, mask, **kwargs):
        loss = self.loss_fn(y_true=y_true, y_pred=y_pred, **kwargs)
        mask = tf.cast(mask, loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
