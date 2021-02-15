
import tensorflow as tf
import pickle
import tensorflow_datasets as tfds


def encode(x, y, tokenizer_src, tokenizer_tar):
    tk_src, tk_tar = tokenizer_src, tokenizer_tar
    lang1 = [tk_src.vocab_size] + tk_src.encode(x.numpy()) + [tk_src.vocab_size + 1]
    lang2 = [tk_tar.vocab_size] + tk_tar.encode(y.numpy()) + [tk_tar.vocab_size + 1]
    return lang1, lang2


def filter_max_length(x, y, max_length):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def get_tokenizers(data, tokenizer_path, lang1, lang2, target_vocab_size=2**13):
    print('changed')
    tokenizers = dict()
    if tokenizer_path is not None:
        for lang in [lang1, lang2]:
            tokenizers[lang] = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
                f'{tokenizer_path}/tkn_{lang}'
            )

    else:
        tokenizers[lang1] = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
              (l1.numpy() for l1, _ in data),
              target_vocab_size=target_vocab_size
          )
        tokenizers[lang2] = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
              (l2.numpy() for _, l2 in data),
              target_vocab_size=target_vocab_size
          )

    return tokenizers


def get_datasets(src, tar,
                 batch_size=64,
                 shuffle_buffer_size=20000,
                 target_vocab_size=2**13,
                 max_seq_length=40,
                 tokenizer_path=None,
                 ):

    print('changed')
    lang1, lang2 = src, tar
    code = f'{src}_to_{tar}'
    try:
        examples, metadata = tfds.load(f'ted_hrlr_translate/{code}', with_info=True,
                                       as_supervised=True)
    except:
        try:
            code = f'{tar}_to_{src}'
            examples, metadata = tfds.load(f'ted_hrlr_translate/{code}', with_info=True,
                                           as_supervised=True)
            lang1, lang2 = tar, src
        except:
            raise ValueError(f'No dataset exists in ted_hrlr_translate for the pair {src}, {tar}')

    train_examples, val_examples = examples['train'], examples['validation']

    tokenizers = get_tokenizers(train_examples, tokenizer_path, lang1, lang2, target_vocab_size)

    def _tf_encode(x, y):
        result_src, result_tar = tf.py_function(
            lambda _x, _y: encode(_x, _y, tokenizers[src], tokenizers[tar]),
            [x, y], [tf.int64, tf.int64]
        )
        result_src.set_shape([None])
        result_tar.set_shape([None])
        return result_src, result_tar

    def _filt_max_length(x, y):
        return filter_max_length(x, y, max_seq_length)

    train_dataset = train_examples.map(_tf_encode)
    train_dataset = train_dataset.filter(_filt_max_length)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).padded_batch(batch_size, padded_shapes=([None], [None]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(_tf_encode)
    val_dataset = val_dataset.filter(_filt_max_length).padded_batch(batch_size, padded_shapes=([None], [None]))

    return train_dataset, val_dataset, tokenizers


def build_datasets(src, tar, config):
    return get_datasets(
        src, tar,
        batch_size=config.data.batch_size,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        target_vocab_size=config.data.target_vocab_size,
        max_seq_length=config.data.max_seq_length,
        tokenizer_path=config.data.tokenizer_path

    )










