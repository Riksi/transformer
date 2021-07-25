import tensorflow as tf

from embedding import PositionalEncoding, ScaledEmbedding
from blocks import Encoder, Decoder

print('hi')


class Transformer(tf.keras.models.Model):
    def __init__(self,
                 num_tokens,
                 num_tgt_tokens,
                 model_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 ff_dim=2048,
                 num_encoder_blocks=6,
                 num_decoder_blocks=6,
                 share_embed_weights=False,
                 share_output_weights=False):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.share_embed_weights = share_embed_weights
        self.share_output_weights = share_output_weights
        self.input_embedding = ScaledEmbedding(num_tokens, model_dim)
        self.enc_pos_encoding = PositionalEncoding(model_dim, dropout=dropout)
        self.dec_pos_encoding = PositionalEncoding(model_dim, dropout=dropout)

        if not self.share_embed_weights:
            self.target_embedding = ScaledEmbedding(num_tgt_tokens, model_dim)

        self.encoder = Encoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_encoder_blocks)

        self.decoder = Decoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_decoder_blocks)
        if not self.share_output_weights:
            self.output_layer = tf.keras.layers.Dense(units=num_tgt_tokens)

    def encode(self, x, src_mask, training=True):
        x = self.input_embedding(x)
        x = self.enc_pos_encoding(x, training=training)
        memory, attn = self.encoder(x, mask=src_mask, training=training)
        return memory, attn

    def decode(self, y, memory, src_mask, tgt_mask, training=True):
        if self.share_embed_weights:
            y = self.input_embedding(y)
        else:
            y = self.target_embedding(y)
        y = self.dec_pos_encoding(y, training=training)
        out, self_attn, attn = self.decoder(y, memory,
                           memory_mask=src_mask,
                           decoder_mask=tgt_mask,
                           training=training)
        if not self.share_output_weights:
            logits = self.output_layer(out)
        else:
            # This works because this is called only after
            # target_embedding is called so the weights will
            # have been created
            if not self.share_embed_weights:
                embed = self.target_embedding
            else:
                embed = self.input_embedding
            logits = tf.matmul(out, embed.weights[0], transpose_b=True)
        return logits, self_attn, attn

    def call(self, x, y, src_mask, tgt_mask, training=True):
        memory, enc_attn = self.encode(x, src_mask, training=training)
        logits, dec_self_attn, dec_attn = self.decode(y, memory,
                        src_mask=src_mask,
                        tgt_mask=tgt_mask,
                        training=training)
        attention = dict(
            enc=enc_attn,
            dec_self=dec_self_attn,
            dec_memory=dec_attn
        )
        return logits, attention


class MultiTaskTransformer(tf.keras.models.Model):
    def __init__(self,
                 num_tokens,
                 num_tgt_tokens,
                 model_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 ff_dim=2048,
                 num_encoder_blocks=6,
                 num_decoder_blocks=6,
                 share_embed_weights=False,
                 share_output_weights=False,
                 num_clf_classes=2):
        super(MultiTaskTransformer, self).__init__()
        self.share_embed_weights = share_embed_weights
        self.share_output_weights = share_output_weights
        self.input_embedding = ScaledEmbedding(num_tokens, model_dim)
        self.enc_pos_encoding = PositionalEncoding(model_dim, dropout=dropout)
        self.dec_pos_encoding = PositionalEncoding(model_dim, dropout=dropout)

        if not self.share_embed_weights:
            self.target_embedding = ScaledEmbedding(num_tgt_tokens, model_dim)

        self.encoder = Encoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_encoder_blocks)

        self.decoder = Decoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_decoder_blocks)
        if not self.share_output_weights:
            self.output_layer = tf.keras.layers.Dense(units=num_tgt_tokens)
        self.final_clf = tf.keras.layers.Dense(
            units=num_clf_classes
        )

    def call(self, x, y, src_mask, tgt_mask, training=True):
        x = self.input_embedding(x)
        x = self.enc_pos_encoding(x, training=training)
        memory = self.encoder(x, mask=src_mask, training=training)

        # [N, K]
        pred_clf = self.final_clf(memory[:, 0])

        if self.share_embed_weights:
            y = self.input_embedding(y)
        else:
            y = self.target_embedding(y)
        y = self.dec_pos_encoding(y, training=training)
        out = self.decoder(y, memory,
                            memory_mask=src_mask,
                            decoder_mask=tgt_mask,
                            training=training)

        if self.share_output_weights:
            logits = self.output_layer(out)
        else:
            # This works because this is called only after
            # target_embedding is called so the weights will
            # have been created
            logits = tf.matmul(out, self.target_embedding.weights[0], transpose_b=True)
        return logits, pred_clf