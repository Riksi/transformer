import tensorflow as tf


def predict_sequence(inputs, model, max_length,
                     start_symbol, end_symbol,
                     pad_masker, future_masker):
    result = tf.ones_like(inputs[:, :1]) * start_symbol
    inputs = tf.concat([tf.ones_like(inputs[:,:1]) * start_symbol,
                        inputs,
                        tf.ones_like(inputs[:,:1]) * end_symbol],
                       axis=1)
    for _ in tf.range(max_length):
        # Ensure that previous value is not pad symbol
        # which means sequence has ended
        not_ended = tf.not_equal(result[:, -1], end_symbol)
        not_ended_idx = tf.where(not_ended)
        # Input the previously predicted symbols
        # for just the incomplete sequences
        # [B', i+1]
        res = tf.boolean_mask(result, not_ended)
        inp = tf.boolean_mask(inputs, not_ended)
        # predictions: [B, i+1, T]
        [predictions] = model(inp, res,
                              src_mask=pad_masker(inp),
                              tgt_mask=future_masker(res),
                              training=False)
        # Just select the last sequence element and the symbol
        # with highest probability
        # [B']
        next_symbol = tf.argmax(predictions[:, -1], axis=-1)
        res_next = tf.scatter_nd(not_ended_idx,
                                next_symbol,
                                tf.shape(result[:, -1], out_type=tf.int64))
        res_next = tf.cast(res_next, result.dtype) + (1 - tf.cast(not_ended, result.dtype)) * end_symbol
        # Update just incomplete sequences
        result = tf.concat([result, res_next[:, None]], axis=-1)

        # If all sequences are done stop
        if tf.reduce_all(tf.equal(next_symbol, end_symbol)):
            break
    return result


def inference(config, model, dataloader):
    for step, (inputs, tar) in enumerate(dataloader):
        preds = predict_sequence(
            inputs, model,
            config.data.max_length, config.data.start_symbol,
            config.data.end_symbol
        )
        break
    return (inputs, tar, preds)