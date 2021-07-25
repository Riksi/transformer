import tensorflow as tf
from attention import MultiHeadAttention


class FeedForward(tf.keras.models.Model):
    def __init__(self, hidden_dim, output_dim, activation='relu'):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim,
                                            activation=activation)
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.dense2(self.dense1(x))
        return x


# class ResidualBlock(tf.keras.models.Model):
#     def __init__(self, sublayer, dropout=0.0):
#         super(ResidualBlock, self).__init__()
#         self.sublayer = sublayer
#         self.use_dropout = dropout > 0
#         if self.use_dropout:
#             self.dropout_layer = tf.keras.layers.Dropout(dropout)
#         self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#
#     def call(self, x, *additional_inputs, training=True):
#         outputs = self.sublayer(x, *additional_inputs, training=training)
#         if self.use_dropout:
#             outputs = self.dropout_layer(outputs, training=training)
#         return self.layer_norm(x + outputs, training=training)


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, dropout=0.0):
        super(ResidualLayer, self).__init__()
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, skip, out, training=True):
        if self.use_dropout:
            out = self.dropout_layer(out, training=training)
        return self.layer_norm(skip + out, training=training)


class EncoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.attn_block = MultiHeadAttention(dim, num_heads)
        self.res1 = ResidualLayer(dropout)
        self.ff_block = FeedForward(hidden_dim=ff_dim, output_dim=dim)
        self.res2 = ResidualLayer(dropout)

    def call(self, query, key, value, mask, training=True):
        x, attn = self.attn_block(query, key, value, mask, training=training)
        skip = x = self.res1(skip=query, out=x, training=training)
        x = self.ff_block(x)
        x = self.res2(skip=skip, out=x, training=training)
        return x, attn


class DecoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0):
        super(DecoderBlock, self).__init__()
        self.self_attn_block = MultiHeadAttention(dim, num_heads)
        self.res = ResidualLayer(dropout=dropout)
        self.memory_block = EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)

    def call(self, query, key, value, decoder_mask, memory_mask, training=True):
        # if not self.skip_attn:
        x, self_attn = self.self_attn_block(query, query, query, decoder_mask, training=training)
        x = self.res(skip=query, out=x, training=training)
        x, attn = self.memory_block(x, key, value, memory_mask, training=training)
        return x, self_attn, attn


class Encoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0):
        super(Encoder, self).__init__()
        self.blocks = [
            EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]

    def call(self, query, mask, training=True):
        attn_maps = []
        for block in self.blocks:
            query, attn = block(
                          query=query, key=query, value=query,
                          mask=mask, training=training)
            attn_maps.append(attn)
        return query, attn_maps


class Decoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0):
        super(Decoder, self).__init__()
        self.blocks = [
            DecoderBlock(dim, ff_dim, num_heads, dropout=dropout)
            for i in range(num_blocks)
        ]

    def call(self, query, memory, decoder_mask, memory_mask, training=True):
        self_attn_maps = []
        memory_attn_maps = []
        for block in self.blocks:
            query, self_attn, memory_attn = block(
                      query=query, key=memory, value=memory,
                      decoder_mask=decoder_mask,
                      memory_mask=memory_mask,
                      training=training)
            self_attn_maps.append(self_attn)
            memory_attn_maps.append(memory_attn)

        return query, self_attn_maps, memory_attn_maps