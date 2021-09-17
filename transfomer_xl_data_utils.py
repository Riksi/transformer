from absl import flags
import numpy as np
import math
import pickle
import json
import tensorflow as tf
import os

# TODO:
# - [ ] Finish dataset creation
# - [x] Finish vocab
# - [ ] Try it on a small dataset to see what happens
#    - [x] Have done upto build_vocab
#    - [x] Now need to try `encode_file`
# - [ ] Go through the steps of dataset creation
#     - [ ] Working on a tiny dataset to see what is happening
#           - it looks like the shards cycle through the text - I don't know if this is ok ...


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        # In all of this I am using settings given for WikiText-103

        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        self.vocab.count_file(
            os.path.join(path, "train.txt")
        )

        self.vocab.build_vocab()

        for fold in ['train', 'valid', 'test']:
            setattr(self, fold,
                    self.vocab.encode_file(os.path.join(path, f"{fold}.txt"), ordered=True))

        # TODO: maybe these should be different?
        self.cutoffs = [] #0, 20000, 40000, 200000] + [len(self.vocab)]

    def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len,
                             num_core_per_host, **kwargs):
        FLAGS = kwargs.get('FLAGS')

        file_names = []
        use_tpu = FLAGS.use_tpu and not (split == "test" and num_core_per_host == 1)

        if use_tpu:
            record_name = "record_info-{}.bsz-{}.tlen-{}.core-{}.json".format(
                split, bsz, tgt_len, num_core_per_host)
        else:
            record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
                split, bsz, tgt_len)

        record_info_path = os.path.join(save_dir, record_name)

        data = getattr(self, split)

        bin_sizes = get_bin_sizes(data, bsz // num_core_per_host, tgt_len, self.cutoffs)
        file_name, num_batch = create_ordered_tfrecords(
            save_dir, split, data, bsz, tgt_len, num_core_per_host,
            self.cutoffs, bin_sizes,
            num_passes=FLAGS.num_passes if split == 'train' and use_tpu else 1,
            use_tpu=use_tpu)
        file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
                "filenames": file_names,
                "bin_sizes": bin_sizes,
                "num_batch": num_batch
            }
            json.dump(record_info, fp)


def get_bin_sizes(data, batch_size, tgt_len, cutoffs, std_mult=[2.5, 2.5, 2.5]):
    # batch_size should be per-core batch_size

    bin_sizes = []

    def _nearest_multiple_of_eight(x):
        # i.e. nearest that is greater than 0
        y = x - x % 8
        return y + 8 if x % 8 >= 4 else max(8, y)

    if cutoffs:
        # approximate number of batches
        num_batch = len(data) // batch_size // tgt_len

        # drop last part and reshape
        data = data[:batch_size * num_batch * tgt_len]
        data = data.reshape(batch_size, num_batch, tgt_len)

        # examples in a batch
        tot = batch_size * tgt_len

        # here it would be
        # [20000, 40000), [40000, 200000), [200000, end)
        # TODO: why is [0, 20000) dropped
        for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):
            mask = (data >= left) * (data < right)
            percents = mask.astype(np.float64).sum(2).sum(0) / tot
            mean = np.mean(percents)
            std = np.std(percents)

            bin_size = int(math.ceil(tgt_len * batch_size * (mean + std_mult[b] * std)))
            bin_size = _nearest_multiple_of_eight(bin_size)
            bin_sizes.append(bin_size)

    return bin_sizes


def batchify(data, batch_size, num_passes):
    if num_passes > 1:
        data_len = len(data)
        double_data = np.concatenate([data, data])
        data_list = []
        # Basically you are concatenating `num_passes` chunks of size `data_len`
        # with random starting points along the data
        # Data is [a, b, x, y, z]
        # -> [a, b, x, y, z, a, b, x, y, z]
        # -> [b, x, y, z, a], [z, a, b, x, y], [x, y, z, a, b], etc.

        for i in range(num_passes):
            start = np.random.randint(0, data_len)
            data_list.append(double_data[start:start+data_len])

        data = np.concatenate(data_list)

    num_step = len(data) // batch_size
    data = data[:batch_size * num_step]
    data = data.reshape((batch_size, num_step))

    return data


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len,
                             num_core_per_host, cutoffs=[], bin_sizes=[],
                             num_passes=1, use_tpu=False):

  if use_tpu:
    file_name = "{}.bsz-{}.tlen-{}.core-{}.tfrecords".format(
        basename, batch_size, tgt_len, num_core_per_host)
  else:
    file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(
        basename, batch_size, tgt_len)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.io.TFRecordWriter(save_path)

  batched_data = batchify(data, batch_size, num_passes)

  num_batch = 0
  # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
  for t in range(0, batched_data.shape[1] - 1, tgt_len):
    cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
    # drop the remainder if use tpu
    if use_tpu and cur_tgt_len < tgt_len:
      break
    if num_batch % 500 == 0:
      print("  processing batch {}".format(num_batch))
    for idx in range(batch_size):
      inputs = batched_data[idx, t:t + cur_tgt_len]
      labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

      # features dict
      feature = {
          "inputs": _int64_feature(inputs),
          "labels": _int64_feature(labels),
      }

      if len(cutoffs) > 0 and use_tpu:
        # validate `bin_sizes` and `cutoffs`
        assert len(cutoffs) - len(bin_sizes) == 2, \
          "len(cutoffs) - len(bin_sizes) != 2"

        # mask for bin 0
        left, right = cutoffs[:2]
        inp_mask = ((inputs >= left) * (inputs < right)).astype(np.float32)
        tgt_mask = ((labels >= left) * (labels < right)).astype(np.float32)

        feature["inp_mask"] = _float_feature(inp_mask)
        feature["tgt_mask"] = _float_feature(tgt_mask)

        # refresh `inp_cnts` and `tgt_cnts` for each TPU core
        if idx % (batch_size // num_core_per_host) == 0:
          inp_cnts = [0] * len(bin_sizes)
          tgt_cnts = [0] * len(bin_sizes)

        head_labels = np.copy(labels)
        inp_pos_per_bin, tgt_pos_per_bin = [], []


        for b, (left, right) in enumerate(zip(cutoffs[1:-1], cutoffs[2:])):

          inp_pos = np.where((inputs >= left) * (inputs < right))[0]
          tgt_pos = np.where((labels >= left) * (labels < right))[0]
          inp_pos_per_bin.append(inp_pos)
          tgt_pos_per_bin.append(tgt_pos)

          # All these target locations have this value
          head_labels[tgt_pos] = cutoffs[1] + b

        feature["head_labels"] = _int64_feature(head_labels)

        # permutation feature
        def _add_perm_feature(feature, pos_per_bin, cnts, prefix):
          # Go through each bin
          for b, pos in enumerate(pos_per_bin):
            idx_tuple = []
            # if bin is not full add to bin (bin_id, num_so_far)
            for p in pos:
              if cnts[b] < bin_sizes[b]:
                idx_tuple.append([p, cnts[b]])
                cnts[b] += 1
              else:
                break

            n_tup = len(idx_tuple)
            tup = np.array(idx_tuple).reshape(n_tup * 2)

            # inp_cnt_<bin_id> is total number in bin
            feature["{}_cnt_{}".format(prefix, b)] = _int64_feature([n_tup])
            # inp_tup_<bin_id> is the list of (bin_id, num_so_far)
            feature["{}_tup_{}".format(prefix, b)] = _int64_feature(tup)

        _add_perm_feature(feature, inp_pos_per_bin, inp_cnts, "inp")
        _add_perm_feature(feature, tgt_pos_per_bin, tgt_cnts, "tgt")

      example = tf.train.Example(features=tf.train.Features(feature=feature))
      record_writer.write(example.SerializeToString())

    num_batch += 1

  record_writer.close()
  print("Done writing {}. batches: {}".format(file_name, num_batch))

  return file_name, num_batch


def get_lm_corpus(data_dir, dataset):
    fn = os.path.join(data_dir, "cache.pkl")

    if tf.io.gfile.exists(fn):
        print("Loading cached dataset")
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)

    else:
        print("Producing dataset...")
        kwargs = dict()
        # I am using settings given for WikiText-103
        kwargs["special"] = ["<eos>"]
        kwargs["lower_case"] = False
        corpus = Corpus(data_dir, dataset, **kwargs)

        print("Saving dataset...")

        with open(fn, "wb") as fp:
            # TODO: why protocol = 2 ?
            pickle.dump(corpus, fp, protocol=2)

        corpus_info = dict(
            vocab_size=len(corpus.vocab),
            cutoffs=corpus.cutoffs,
            dataset=corpus.dataset
        )

        with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
            json.dump(corpus_info, fp)

    return corpus


