import tensorflow as tf


class ScaledEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_tokens, dim):
        super(ScaledEmbedding, self).__init__()
        self.embed = tf.keras.layers.Embedding(
                input_dim=num_tokens,
                output_dim=dim
        )
        self.dim = tf.cast(dim, tf.float32)

    def call(self, x):
        return tf.sqrt(self.dim) * self.embed(x)


class PositionalEncoding(tf.keras.models.Model):
    def __init__(self, dim, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        # (D / 2,)
        self.range = tf.range(0, dim, 2)
        self.dim = tf.cast( 1 / (10000 ** (self.range / dim)), tf.float32)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=True):
        # x: (B, N, D)
        # (N,)
        length = tf.shape(x)[-2]
        pos = tf.cast(tf.range(length), tf.float32)
        # (1, N) / (D / 2, 1) -> (D / 2, N)
        inp = pos[None] * self.dim[:, None]
        sine = tf.sin(inp)
        cos = tf.cos(inp)
        # (D, N)
        enc = tf.dynamic_stitch(
            indices=[self.range, self.range + 1],
            data=[sine, cos]
        )
        # (N, D)
        enc = tf.transpose(enc, (1, 0))[None]

        if self.use_dropout:
            return self.dropout_layer(x + enc, training=training)
        return x + enc


# class RelativePositionalEncoding(tf.keras.layers.Layer)
#     def __init__(self, dim, dropout=0.0):


