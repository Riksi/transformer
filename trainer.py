import tensorflow as tf
from loss import smooth_labels
print('changedz')

# def accuracy_function(real, pred):
#     accuracies = tf.equal(real, tf.argmax(pred, axis=2))
#
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     accuracies = tf.math.logical_and(mask, accuracies)
#
#     accuracies = tf.cast(accuracies, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def split_target(tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    return tar_inp, tar_real


def accuracy_function(real, pred, pad_mask):
    pred_ids = tf.argmax(pred, axis=2)
    accuracies = tf.cast(tf.equal(tf.cast(real, pred_ids.dtype), pred_ids), tf.float32)
    mask = tf.cast(pad_mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies * mask)/tf.reduce_sum(mask)


class Trainer(tf.Module):
    def __init__(self,
                 config,
                 model: tf.keras.models.Model,
                 pad_masker,
                 future_masker,
                 loss_function,
                 optim):
        self.config = config
        self.model = model
        self.pad_masker = pad_masker
        self.future_masker = future_masker
        self.loss_function = loss_function
        self.optim = optim
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ))
    def train_step(self, inp, tar):
        tar_inp, tar_real = split_target(tar)
        tar_pad_mask = self.pad_masker(tar_real)[:, 0, 0, :]

        with tf.GradientTape() as tape:
            _, predictions, _ = self.model(
                inp, tar_inp,
                src_mask=self.pad_masker(inp),
                tgt_mask=self.future_masker(tar_inp),
                training=True
            )

            tar_onehot = tf.one_hot(tar_real, depth=tf.shape(predictions)[-1])

            if self.config.smoothing_prob > 0:
                labels = smooth_labels(tar_onehot, self.config.smoothing_prob)
            else:
                labels = tar_onehot

            loss = self.loss_function(
                y_true=labels,
                y_pred=predictions,
                mask=tar_pad_mask
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions, tar_pad_mask))

    @tf.function(input_signature=(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32)
    ))
    def valid_step(self, inp, tar):
        tar_inp, tar_real = split_target(tar)
        tar_pad_mask = self.pad_masker(tar_real)[:, 0, 0, :]

        _, predictions, _ = self.model(
                inp, tar_inp,
                src_mask=self.pad_masker(inp),
                tgt_mask=self.future_masker(tar_inp),
                training=False
            )

        loss = self.loss_function(
            y_true=tf.one_hot(tar_real, depth=tf.shape(predictions)[-1]),
            y_pred=predictions,
            mask=tar_pad_mask
        )

        self.val_loss(loss)
        self.val_accuracy(accuracy_function(tar_real, predictions, tar_pad_mask))



class MultiTaskTrainer(object):
    def __init__(self,
                 model: tf.keras.models.Model,
                 pad_masker,
                 future_masker,
                 seq_loss_function,
                 clf_loss_function,
                 optim,
                 loss_weights=None):
        print('changed')
        self.model = model
        self.pad_masker = pad_masker
        self.future_masker = future_masker
        self.seq_loss_function = seq_loss_function
        self.clf_loss_function = clf_loss_function
        self.optim = optim
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        for mode in ['clf', 'seq']:
            for l in ['loss', 'acc']:
                for split in ['train', 'val']:
                    m = f'{split}_{mode}_{l}'
                    setattr(self, m, tf.keras.metrics.Mean(name=m))

        self.loss_weights = loss_weights

    def train_step(self, inp, seq_tar, clf_tar):
        tar_inp = seq_tar[:, :-1]
        tar_real = seq_tar[:, 1:]

        with tf.GradientTape() as tape:
            seq_preds, clf_preds = self.model(
                inp, tar_inp,
                src_mask=self.pad_masker(inp),
                tgt_mask=self.future_masker(tar_inp),
                training=True
            )

            seq_loss = self.seq_loss_function(
                y_true=tf.one_hot(tar_real, depth=tf.shape(seq_preds)[-1]),
                y_pred=seq_preds,
                mask=tf.cast(self.pad_masker(tar_real)[:, 0, 0, :], tf.float32)
            )

            clf_loss = self.clf_loss_function(
                y_true=clf_tar,
                y_pred=clf_preds
            )
            if self.loss_weights is not None:
                loss = clf_loss * self.loss_weights['clf'] + seq_loss * self.loss_weights['seq']
            else:
                loss = clf_loss + seq_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)



        self.train_seq_loss(seq_loss)
        self.train_seq_acc(accuracy_function(tar_real, seq_preds))

        self.train_clf_loss(clf_loss)
        self.train_clf_acc(

            tf.reduce_mean(
                tf.cast(

                    tf.equal(
                        tf.argmax(clf_preds, axis=-1),
                        clf_tar
                    ),
                    tf.float32
                )

            )

        )

    def valid_step(self, inp, seq_tar, clf_tar):
        tar_inp = seq_tar[:, :-1]
        tar_real = seq_tar[:, 1:]

        seq_preds, clf_preds = self.model(
            inp, tar_inp,
            src_mask=self.pad_masker(inp),
            tgt_mask=self.future_masker(tar_inp),
            training=True
        )

        seq_loss = self.seq_loss_function(
            y_true=tf.one_hot(tar_real, depth=tf.shape(seq_preds)[-1]),
            y_pred=seq_preds,
            mask=tf.cast(self.pad_masker(tar_real)[:, 0, 0, :], tf.float32)
        )

        clf_loss = self.clf_loss_function(
            y_true=clf_tar,
            y_pred=clf_preds
        )

        if self.loss_weights is not None:
            loss = clf_loss * self.loss_weights['clf'] + seq_loss * self.loss_weights['seq']
        else:
            loss = clf_loss + seq_loss

        self.val_loss(loss)

        self.val_seq_loss(seq_loss)
        self.val_seq_acc(accuracy_function(tar_real, seq_preds))

        self.val_clf_loss(clf_loss)
        self.val_clf_acc(

            tf.reduce_mean(
                tf.cast(

                    tf.equal(
                        tf.argmax(clf_preds, axis=-1),
                        clf_tar
                    ),
                    tf.float32
                )

            )

        )


