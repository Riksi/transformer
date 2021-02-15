import tensorflow as tf


class MaskedLoss(object):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, y_true, y_pred, mask):
        loss = self.loss_fn(y_true=y_true, y_pred=y_pred)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)