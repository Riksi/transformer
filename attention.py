import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask, inf=1e9):
    # x: (query=(B, H, N_q, d), key=(B, H, N_kv, d), value=(B, H, N_kv, d))
    # mask: (B, 1, N_q, N_kv) or (B, 1, 1, N_kv)
    dim = tf.cast(tf.shape(query)[-1], tf.float32)
    # (B, H, N_q, N_kv)
    alpha_term = tf.matmul(query, key, transpose_b=True) / tf.sqrt(dim)
    # (B, H, N_q, N_kv)
    alpha_term = tf.where(mask, alpha_term, -inf)
    alpha = tf.nn.softmax(alpha_term)
    # (B, H, N_q, N_kv) @ (B, H, N_kv, d') -> (B, H, N_q, d')
    out = tf.matmul(alpha, value)

    return out, alpha


class MultiHeadAttention(tf.keras.models.Model):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.transform_query, self.transform_key, self.transform_value = [
            *(tf.keras.layers.Dense(units=dim) for _ in range(3))
        ]
        self.transform_out = tf.keras.layers.Dense(units=dim)

    def split_heads(self, x):
        # x: (B, N, d)
        # (B, N, h, d//h)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self.dim // self.num_heads))
        # (B, h, N, d//h)
        x = tf.transpose(x, (0, 2, 1, 3))
        return x

    def merge_heads(self, x):
        # x: (B, h, N, d//h)
        # (B, N, h, d//h)
        x = tf.transpose(x, (0, 2, 1, 3))
        # (B, N, d)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.dim))
        return x

    def call(self, query, key, value, mask):
        # (query=(B, N_q, d), key=(B, N_k, d), value=(B, N_v, d))
        query = self.transform_query(query)
        key = self.transform_key(key)
        value = self.transform_value(value)
        # (query=(B, h, N_q, d//h), key=(B, h, N_k, d//h), value=(B, h, N_v, d//h))
        query, key, value = (self.split_heads(i) for i in [query, key, value])
        # (B, h, N_q, d)
        x, attn = scaled_dot_product_attention(query, key, value, mask)

        x = self.merge_heads(x)
        x = self.transform_out(x)

        return x, attn


class NaiveRelativeMultiHeadAttention(tf.keras.models.Model):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        # TODO: should use bias be false for these???
        self.transform_query, self.transform_key, self.transform_value, self.transform_key_rel = (
            tf.keras.layers.Dense(units=dim) for _ in range(4))

        self.weight_key = tf.keras.layers.Dense(units=1, use_bias=False)
        self.weight_rel = tf.keras.layers.Dense(units=1, use_bias=False)

        self.transform_out = tf.keras.layers.Dense(units=dim)

    def split_heads(self, x):
        # x: (B, N, d)
        # (B, N, h, d//h)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.num_heads, self.dim // self.num_heads))
        # (B, h, N, d//h)
        x = tf.transpose(x, (0, 2, 1, 3))
        return x

    def merge_heads(self, x):
        # x: (B, h, N, d//h)
        # (B, N, h, d//h)
        x = tf.transpose(x, (0, 2, 1, 3))
        # (B, N, d)
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.dim))
        return x

    def call(self, query, key, value, mask, rel):
        # (query=(B, N_q, d), key=(B, N_k, d), value=(B, N_v, d))
        # rel: (N_q + N_k, d)
        query = self.transform_query(query)
        key = self.transform_key(key)
        value = self.transform_value(value)
        # (query=(B, h, N_q, d//h), key=(B, h, N_k, d//h), value=(B, h, N_v, d//h))
        query, key, value = (self.split_heads(i) for i in [query, key, value])
        # (B, h, N_q, d)

        # (N_q + N_k, d)
        # TODO: show this be the same as `transform_key_rel`
        rel = self.transform_key_rel(rel[::-1])

        # The matrices will be aligned so that each query is multiplied
        # with its corresponding set of relative encodings
        # (B, h, N_q, d//h), (h, N_q + N_kv, d//h)
        # -> (B, h, N_q, d//h),
        #       (h, d//h, N_q + N_kv)
        # -> (B, h, N_q, N_q + N_kv) -> (B, h, N_q, N_q + N_kv)
        query_rel_term = tf.matmul(query, rel, transpose_b=True)

        N_kv = tf.shape(key)[1]
        N_r = tf.shape(rel)[1]

        # We want the left N_kv + 1, N_kv + 2, etc. values of query_rel_mask but we want to
        # place them in the right N_kv + 1, N_kv + 2, etc. values of a new tensor

        # Initially mask the right N_kv + 1, N_kv + 2, etc. values
        # (N_q, N_q + N_kv)
        # N_r - N_kv = (N_q + N_kv) - N_kv = N_q
        non_zero_mask = tf.sequence_mask(
            tf.range(N_kv + 1, N_r + 1), N_r
        )

        # (B, h, N_q, N_q + N_kv)
        non_zero_mask = tf.broadcast_to(non_zero_mask[None, None], tf.shape(query_rel_term))

        # The True positions give the indices for the result
        non_zero_idx = tf.where(non_zero_mask)

        # Reverse the mask  since we want to keepp the leftmost values
        # Note that the the indices are row by row and the order of values
        # selected is the same as the order of the indices
        non_zero_terms = query_rel_term[non_zero_mask[..., ::-1]]

        query_rel_term = tf.scatter_nd(
            indices=non_zero_idx,
            updates=non_zero_terms,
            shape=tf.shape(query_rel_term)
        )
        query_key_term = tf.matmul(query, key, transpose_b=True)

        # (B, h, N_kv, d//h) -> (B, h, N_kv, 1)
        # -> (B, h, 1, N_kv, 1) -> (B, h, 1, N_kv)
        key_term = tf.squeeze(self.weight_key(key)[:, :, None], dim=-1)

        # (h, N_q, N_kv, d//h) -> (h, N_q, N_kv, 1)
        # -> (h, N_q + N_kv, 1) -> (h, N_q + N_kv)
        rel_term = tf.squeeze(self.weight_rel(rel), dim=-1)

        # (h, N_q, N_q + N_kv)
        rel_term = tf.tile(rel_term[:, None], [1, tf.shape(query)[1], 1])

        # non_zero_mask is repeated for each batch element and head so we can
        # just reuse it by selecting the first element of it
        rel_term = tf.scatter_nd(
            indices=tf.where(non_zero_mask[0]),
            updates=rel_term[non_zero_mask[0][..., ::-1]],
            shape=tf.shape(rel_term)
        )

        # (B, h, N_q, N_kv) + (B, h, N_q, N_kv) + (B, h, 1, N_kv) + (h, N_q, N_kv)
        # = (B, h, N_q, N_kv)
        attn_raw = query_key_term + query_rel_term + key_term + rel_term

        # TODO: this step is not shown explicitly
        #  so not sure if it is needed
        attn_raw = attn_raw / tf.sqrt(self.dim)

        # (B, h, N_q, N_kv)
        attn_raw = tf.where(mask, attn_raw, 1e9)

        # (B, h, N_q, N_kv)
        attn = tf.nn.softmax(attn_raw)

        # (B, h, N_q, N_kv) @ (B, h, N_kv, d//h) -> (B, h, N_q, d//h)
        x = tf.matmul(attn, value)

        # (B, N_q, d)
        x = self.merge_heads(x)

        # (B, N_q, d)
        x = self.transform_out(x)
        return x



        # # (N_q, N_k, h, d//h)
        # rel = tf.reshape(rel, tf.concat([tf.shape(rel)[:2],
        #                                  [self.num_heads, self.dim // self.num_heads]], axis=0))
        # # (h, N_q, N_k, d//h)
        # rel = tf.transpose(rel, (2, 0, 1, 3))

        # # The matrices will be aligned so that each query is multiplied
        # # with its corresponding set of relative encodings
        # # (B, h, N_q, 1, d//h), (h, N_q, N_kv, d//h)
        # # -> (B, h, N_q, 1,    d//h),
        # #       (h, N_q, d//h, N_kv)
        # # -> (B, h, N_q, 1,    N_kv) -> (B, h, N_q, N_kv)
        # query_rel_term = tf.squeeze(
        #     tf.matmul(query[..., None, :], rel, transpose_b=True),
        #     axis=-2
        # )