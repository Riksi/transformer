import tensorflow as tf
import math

from embedding import PositionalEncoding, ScaledEmbedding
from blocks import Encoder


class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, decay_steps, warmup_steps, alpha):
        self._lr = lr
        self._warmup_steps = warmup_steps
        # offset by warmup_steps
        self._decay_steps = decay_steps - warmup_steps
        self._alpha = alpha

    def get_config(self):
        return dict(lr=self._lr,
                    decay_steps=self._decay_steps,
                    warmup_steps=self._warmup_steps,
                    alpha=self._alpha)

    def decayed_learning_rate(self, step):
        # offset by warmup_steps
        step = step - self._warmup_steps
        decay_steps = self._decay_steps
        alpha = self._alpha
        step = tf.minimum(step, decay_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return self._lr * decayed

    def __call__(self, step):
        if step < self._warmup_steps:
            if self._warmup_steps == 0:
                lr = 0.0
            else:
                lr = step / self._warmup_steps * self._lr
        else:
            lr = self.decayed_learning_rate(step)
        return lr


class TransformerXL(tf.keras.models.Model):
    def __init__(self,
                 num_tokens,
                 model_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 ff_dim=2048,
                 num_encoder_blocks=6,
                 share_embed_weights=False,
                 share_output_weights=False):
        super(TransformerXL, self).__init__()
        self.share_embed_weights = share_embed_weights
        self.share_output_weights = share_output_weights
        self.input_embedding = ScaledEmbedding(num_tokens, model_dim)
        self.enc_pos_encoding = PositionalEncoding(model_dim, dropout=dropout)


        self.encoder = Encoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_encoder_blocks)

    def call(self, inputs, mask=None, training=True):
        x = self.input_embedding(inputs)
        x = self.enc_pos_encoding(x, training=training)
        logits, cache = self.encoder(x, mask=mask, training=training)
        return logits, cache

    def train_step(self, data):
        with tf.GradientTape() as tape:
            logits, cache = self.call(data.inputs, data.mask)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data.targets)
            # TODO: add masking
            loss = tf.reduce_mean(loss)

        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)

        clipped, gnorm = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(clipped, variables))

        return loss, cache
