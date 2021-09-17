import tensorflow as tf
from attention import MultiHeadAttention


class FeedForward(tf.keras.models.Model):
    def __init__(self, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim,
                                            activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.dense2(self.dense1(x))
        return x


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, sublayer, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.sublayer = sublayer
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, *additional_inputs, training=True):
        outputs = self.sublayer(x, *additional_inputs, training=training)
        if self.use_dropout:
            outputs = self.dropout_layer(outputs, training=training)
        return self.layer_norm(x + outputs, training=training)


class EncoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.attn_block = ResidualBlock(
            MultiHeadAttention(dim, num_heads),
            dropout=dropout
        )
        self.ff_block = ResidualBlock(
            FeedForward(hidden_dim=ff_dim, output_dim=dim),
            dropout=dropout
        )

    def call(self, query, key, value, mask, training=True):
        out = self.attn_block(query, key, value, mask, training=training)
        out = self.ff_block(out, training=training)
        return out


class DecoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0,
                 skip_attn=False):
        super(DecoderBlock, self).__init__()
        self.skip_attn = skip_attn
        if not skip_attn:
            self.self_attn_block = ResidualBlock(
                MultiHeadAttention(dim, num_heads),
                dropout=dropout
            )
        self.memory_block = EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)

    def call(self, query, key, value, decoder_mask, memory_mask, training=True):
        if not self.skip_attn:
            query = self.self_attn_block(query, query, query, decoder_mask, training=training)
        out = self.memory_block(query, key, value, memory_mask, training=training)
        return out


class Encoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0):
        super(Encoder, self).__init__()
        self.blocks = [
            EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]

    def call(self, query, mask, training=True):
        for block in self.blocks:
            query = block(query=query, key=query, value=query,
                          mask=mask, training=training)
        return query


class Decoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0,
                 skip_first_attn=False):
        super(Decoder, self).__init__()
        self.blocks = [
            DecoderBlock(dim, ff_dim, num_heads, dropout=dropout,
                         skip_attn=(i == 0) and skip_first_attn)
            for i in range(num_blocks)
        ]

    def call(self, query, memory, decoder_mask, memory_mask, training=True):
        out = query
        for block in self.blocks:
            out = block(query=out, key=memory, value=memory,
                      decoder_mask=decoder_mask,
                      memory_mask=memory_mask,
                      training=training)

        return out