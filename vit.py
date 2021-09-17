import tensorflow as tf
import numpy as np
import pickle
from easydict import EasyDict
from typing import List, Union, Callable

import json
from blocks import MultiHeadAttention, FeedForward
import os
import attr
import datetime
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from functools import partial

from randaugment import distort_image_with_randaugment

print('updated-2')

# TODO:
# - [x] Add learned PE
# - [x] Implement modified attention and MLP blocks - refer to notes and equations 1-4
# - [x] Where to add dropout
#       > Dropout, when used, is applied after
# every dense layer except for the the qkv-projections and directly after adding positional- to patch
# embeddings.
#    - [x] What about the output Dense Layer? - no dropout here
#    - [x] Note that in the basic Transformer dropout is not used for every Dense layer
#          but only after both layers of the MLP and the attention block
#          - look at what they do in the official version - used after each layer
# - [x] What are the settings for LayerNorm ?  --- uses default settings
# - [x] Implement encoder
# - [x] Add learning rate
# - [x] Look up the details for other things as mentioned above
# - [x] Add dataloader for CIFAR-10 and get it running
# - [x] CIFAR-10 bounding box
# - [x] Log lr

# TODO:
# NEXT STEPS
# - [x] Look up the parameters from the codebase - I think they have some dataframe for this
#       - has no info for the desired cases
# - [x] Add remaining things for mixup
# - [x] What is the validation split ? - Not mentioned for Resisc45
# - [x] Add stochastic depth
# - [ ] Try if randaug works but not before you have done everything above
# -------------------------------------------------------------------------------------
# - [ ] I wonder how the encodings are interpolated
#   - DO NOT TRY THIS FOR THE UNTIL YOU HAVE TRAINED A MODEL AS IT WILL BE A HUGE TIME SINK

@attr.s(auto_attribs=True)
class Args(object):
    save_path: str
    batch_size: int
    num_classes: int
    grad_clip: Union[float, None] = None
    dataset_name: str = 'cifar10'
    debug_train: bool = False
    debug_val: bool = False
    debug_train_steps: int = 2
    debug_val_steps: int = 2
    data_dir: Union[str, None] = None
    arch: str = 'vit'
    image_size: int = 32
    log_every: int = 100
    patch_size: int = 16
    epochs: int = 300
    # All models are trained with a batch size of 4096 and learning
    # rate warmup of 10k steps. For ImageNet we found it beneficial to additionally apply gradient
    # clipping at global norm 1.
    last_res_prob: float = 1.
    warmup_steps: int = 10000
    base_lr: float = 3e-3
    end_lr: float = 1e-5
    weight_decay: float = 0.3
    dropout: float = 0.1
    mixup_alpha: Union[float, None] = None
    lr_schedule: Union[str, None] = 'cosine'
    resume: bool = False
    resume_path: str = None

    # This is for ViT-S
    model_dim: int = 384
    ff_dim: int = 768
    num_heads: int = 6
    num_layers: int = 12
    # > The only difference to the original ViT
    # > model [10] in our paper is that we drop the hidden layer in the head,
    # > as empirically it does not lead to
    # > more accurate models and often results in optimization instabilities.
    representation_size: Union[int, None] = None

    randaug_num_layers: int = 0
    randaug_magnitude: int = 0

    encoding_size: int = attr.ib(
        default=attr.Factory(
            lambda self: (self.image_size // self.patch_size) ** 2,
            takes_self=True
        )
    )

    model_name: str = attr.ib(
        default=attr.Factory(
            lambda self: f"{self.arch}_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}",
            takes_self=True
        )
    )
    model_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.save_path, self.model_name),
            takes_self=True
        )
    )
    ckpt_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.model_path, 'ckpts'),
            takes_self=True
        )
    )
    tbd_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.model_path, 'tbd'),
            takes_self=True
        )
    )



@tf.function
def get_acc(logits, labels, in_top=(1,), reduce=True):
    # logits: [B, Z]
    # labels: [B] or [B, N] e.g. for mixup
    # [B, Z] -> [B, T]
    # Force it to have 2 dimensions
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
    max_top = max(in_top)
    top = tf.nn.top_k(logits, max_top).indices
    res = dict()

    # [A, T]
    # exclude after the first in_top[i]
    top_mask = tf.sequence_mask(
        in_top, max_top
    )

    # [1, A, T], [B, 1, T] -> [B, A, T]
    top = tf.where(top_mask[None], top[:, None], -1)
    # [B, N]
    labels = tf.cast(labels, top.dtype)
    # [B, A, T, 1], [B, 1, 1, N] -> [B, A]
    contains = tf.reduce_any(tf.equal(top[..., None], labels[:, None, None]), axis=(-2, -1))

    if reduce:
        # [A]
        value = tf.reduce_mean(tf.cast(contains, tf.float32), axis=0)
    else:
        # [A, B]
        value = tf.transpose(tf.cast(contains, tf.float32), (1, 0))

    for idx, i in enumerate(in_top):
        res[f'acc@{i}'] = value[idx]

    return res


class MLP(FeedForward):
    def __init__(self, *args, dropout=0.0, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        if dropout > 0.0:
            print('MLP: using dropout')
            self.use_dropout = True
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False

        self.dense1.kernel_initializer = tf.initializers.random_normal(stddev=1e-6)
        self.dense2.kernel_initializer = tf.initializers.random_normal(stddev=1e-6)

    def call(self, x, training=None):
        x = self.dense1(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        return self.dense2(x)


class ViTBlock(tf.keras.models.Model):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.0, keep_res_probs=None):
        super(ViTBlock, self).__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.attn = MultiHeadAttention(model_dim, num_heads=num_heads)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        # ODOT: Confirm this is the structure of MLP - confirmed
        # (difference from before was gelu instead of relu)
        self.mlp = MLP(hidden_dim=ff_dim, output_dim=model_dim, activation=tf.nn.gelu, dropout=dropout)
        if dropout > 0.:
            print('ViTBlock: using dropout')
            self.use_dropout = True
            self.dropout_layer1 = tf.keras.layers.Dropout(dropout)
            self.dropout_layer2 = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False

        self.stochastic_depth = keep_res_probs is not None
        if self.stochastic_depth:
            print('ViTBlock: using stochastic depth')
        self.keep_res_probs = keep_res_probs

    def add_res(self, skip, x, prob, training):
        if not self.stochastic_depth:
            return skip + x
        if training:
            return skip + tf.cast(tf.random.uniform([]) < prob, tf.float32) * x
        return skip + prob * x

    def call(self, inputs, training=None, **kwargs):
        skip = inputs
        x = self.layer_norm_1(inputs, training=training)
        x, attn = self.attn(x, x, x)
        if self.use_dropout:
            x = self.dropout_layer1(x, training=training)

        skip = x = self.add_res(skip, x,
                                self.keep_res_probs[0] if self.stochastic_depth else None,
                                training=training)

        x = self.mlp(self.layer_norm_2(x, training=training))
        if self.use_dropout:
            x = self.dropout_layer2(x, training=training)

        x = self.add_res(skip, x,
                         self.keep_res_probs[1] if self.stochastic_depth else None,
                         training=training)

        return x, attn


class Encoder(tf.keras.models.Model):
    def __init__(self, model_dim, num_heads, ff_dim, num_blocks, dropout=0.0, last_res_prob=1.):
        super(Encoder, self).__init__()
        blocks = []

        keep_prob_factor = 1.
        if last_res_prob < 1.:
            print('Encoder: using stochastic depth')
            keep_prob_factor = last_res_prob / num_blocks

        for idx in range(num_blocks):
            # Figure shows five blocks and probability for fifth block is denoted as p5
            # Explicitly make 1. if keep_res_prob is 1.
            if last_res_prob < 1.:
                keep_res_probs = list(1 - np.multiply([2 * idx, 2 * idx + 1], keep_prob_factor))
            else:
                keep_res_probs = None

            block = ViTBlock(model_dim=model_dim,
                             num_heads=num_heads,
                             ff_dim=ff_dim,
                             dropout=dropout,
                             keep_res_probs=keep_res_probs)
            blocks.append(block)

        self.blocks = blocks

    def call(self, inputs, mask=None, training=None):
        attn_maps = []
        query = inputs
        for block in self.blocks:
            query, attn = block(query, mask=mask, training=training)
            attn_maps.append(attn)
        return query, attn_maps


class ImageEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, model_dim, dropout=0.):
        super(ImageEmbedding, self).__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(units=self.model_dim, use_bias=False)
        if dropout > 0.:
            print('ImageEmbedding: using dropout')
            self.use_dropout = True
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False
        self.cls_token = tf.Variable(tf.zeros([self.model_dim]))

    def call(self, inputs, pos_encoding, training=None):
        # inputs: [B, H, W, F]
        shape = tf.shape(inputs)
        # Note for F we need the channel dim
        B, H, W, F = shape[0], shape[1], shape[2], inputs.shape[3]
        inputs = tf.reshape(inputs, [B, H//self.patch_size, self.patch_size,
                                        W//self.patch_size, self.patch_size, F])
        # [B, H, W, P, P, F]
        inputs = tf.transpose(inputs, [0, 1, 3, 2, 4, 5])
        # [B, T=H*W/P^2, P^2*F]
        inputs = tf.reshape(inputs, [B, -1, self.patch_size**2 * F])
        # [B, T, D]
        embed = self.projection(inputs)

        # [B, T + 1, D]
        embed = tf.concat(
            [tf.ones_like(embed[:, :1]) * self.cls_token, embed], axis=1
        )

        embed = embed + pos_encoding

        if self.use_dropout:
            embed = self.dropout_layer(embed, training=training)

        return embed


class ViT(tf.keras.models.Model):
    def __init__(self,
                 writers,
                 optimizer,
                 args: Args,
                 # patch_size,
                 # num_classes,
                 # grad_clip=None,
                 # encoding_size=(224 // 16)**2,
                 # model_dim=768,
                 # num_heads=12,
                 # dropout=0.1,
                 # ff_dim=3072,
                 # num_layers=12,
                 # fine_tune=False
                 ):
        super(ViT, self).__init__()
        self.use_mixup = args.mixup_alpha is not None
        if self.use_mixup:
            print('ViT: using mixup')
            self.beta_dist = tfp.distributions.Beta(args.mixup_alpha, args.mixup_alpha)

        self.num_classes = args.num_classes
        self.model_dim = args.model_dim
        self.input_embedding = ImageEmbedding(args.patch_size, args.model_dim, dropout=args.dropout)
        self.pos_encoding = tf.Variable(
            initial_value=tf.random.normal(shape=[args.encoding_size + 1, args.model_dim],
                                           stddev=0.02)
        )
        self.grad_clip = args.grad_clip
        self.encoder = Encoder(
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_blocks=args.num_layers,
            dropout=args.dropout,
            last_res_prob=args.last_res_prob
        )

        self.layer_norm = tf.keras.layers.LayerNormalization()
        # > The classification head is implemented by a MLP
        # with one hidden layer at pre-training
        # time and by a single linear layer at fine-tuning time.

        # Note that representation_size is not included in the default config
        # > The only difference to the original ViT
        # > model [10] in our paper is that we drop the hidden layer in the head,
        # > as empirically it does not lead to
        # > more accurate models and often results in optimization instabilities

        if args.representation_size is not None:
            self.mlp_head = tf.keras.models.Sequential(
                [
                    #TODO: confirm if units=model_dim
                    tf.keras.layers.Dense(units=args.representation_size, activation='tanh'),
                    tf.keras.layers.Dense(units=args.num_classes,
                                          kernel_initializer=tf.initializers.zeros)
                 ]
            )
        else:
            self.mlp_head = tf.keras.layers.Dense(units=args.num_classes,
                                                  kernel_initializer=tf.initializers.zeros)

        self.optimizer = optimizer
        self.writers = writers
        self.epochs_done = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.best_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.best_acc = tf.Variable(0., dtype=tf.float32, trainable=False)



    @tf.function
    def call(self, x, mask=None, training=True):
        # x: [B, H, W, F]
        # [B, T=H*W/P^2, D=P^2*F], [T, D_model]
        x = self.input_embedding(x, self.pos_encoding, training=training)
        # [B, T, D]
        x, attn = self.encoder(x, training=training, mask=mask)
        x = self.layer_norm(x, training=training)
        # [B, D]
        cls_feature = x[:, 0]
        # [B, Z]
        logits = self.mlp_head(cls_feature)
        return logits, x, attn

    @tf.function
    def get_losses_for(self, data, training=None):
        # No mask for now
        logits, x, attn = self.call(data['image'], training=training)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data['label_onehot'])
        )
        acc = get_acc(logits, data['label'], (1, 5))
        return dict(
            acc, loss=loss
        )

    @tf.function
    def train_step(self, data):

        if self.use_mixup:
            image = data['image']
            label = data['label']
            label_onehot = data['label_onehot']

            shuffle_idx = tf.random.shuffle(tf.range(tf.shape(image)[0]))

            lmd = self.beta_dist.sample([])

            image = image * (1 - lmd) + tf.gather(image, shuffle_idx) * lmd

            label_onehot = label_onehot * (1 - lmd) + tf.gather(label_onehot, shuffle_idx) * lmd
            label = tf.stack([label, tf.gather(label, shuffle_idx)], axis=-1)

            data = {"image": image, "label": label, "label_onehot": label_onehot}

        with tf.GradientTape() as tape:
            result = self.get_losses_for(data, training=True)
        grads = tape.gradient(result['loss'], self.trainable_variables)
        if self.grad_clip is not None:
            grads, gnorm = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return result

    @tf.function
    def evaluate(self, dataset):
        count = tf.constant(0.)
        loss = tf.constant(0.)
        acc_1 = tf.constant(0.)
        acc_5 = tf.constant(0.)

        for idx, batch in enumerate(dataset):
            result = self.get_losses_for(batch, training=False)
            batch_size = tf.cast(tf.shape(batch['image'])[0], tf.float32)
            count += batch_size
            loss += result['loss'] * batch_size
            acc_1 += result['acc@1'] * batch_size
            acc_5 += result['acc@5'] * batch_size

        log = {'loss': loss, 'acc@1': acc_1, 'acc@5': acc_5}
        log = {metric: value / count for metric, value in log.items()}

        improved = tf.constant(False)
        if log['acc@1'] > self.best_acc:
            improved = tf.constant(True)
            self.best_acc.assign(log['acc@1'])
            self.best_epoch.assign(self.epochs_done)

        for metric, value in log.items():
            with self.writers['val'].as_default():
                # -1 to align it with train
                tf.summary.scalar(name=metric, data=value, step=tf.cast(self.global_step - 1, tf.int64))

        return log, improved

    #tf.function
    def train_fn(self, dataset, log_every):
        steps_per_epoch = tf.cast(len(dataset), tf.int32)
        log_steps = steps_per_epoch // log_every + 1

        loss = tf.TensorArray(size=log_steps, dtype=tf.float32)
        acc_1 = tf.TensorArray(size=log_steps, dtype=tf.float32)
        acc_5 = tf.TensorArray(size=log_steps, dtype=tf.float32)

        log_idx = 0

        for idx, inputs in enumerate(dataset):

            idx = tf.cast(idx, tf.int32)
            result = self.train_step(inputs)

            if tf.equal(idx % log_every, 0):
                loss = loss.write(log_idx, result['loss'])
                acc_1 = acc_1.write(log_idx, result['acc@1'])
                acc_5 = acc_5.write(log_idx, result['acc@5'])

                log_idx += 1

                tf.print('Epoch', self.epochs_done,
                         '|', 'iteration', idx,
                         ', loss=', result['loss'],
                         ', acc@1=', result['acc@1'],
                         ', acc@5=', result['acc@5'], summarize=100)

                for metric, value in result.items():
                    with self.writers['train'].as_default():
                        with tf.name_scope(''):
                            tf.summary.scalar(
                                name=metric, data=value, step=tf.cast(self.global_step, tf.int64)
                            )

            self.global_step.assign_add(1)

        with self.writers['lr'].as_default():
            # NOTE this assumes the optimizer scheduler does not store state
            # so that if it is called it won't affect how it behaves next time
            if isinstance(self.optimizer.lr, Callable):
                lr = self.optimizer.lr(self.global_step)
            else:
                lr = self.optimizer.lr
            tf.summary.scalar(name='lr',
                              data=lr,
                              step=tf.cast(self.epochs_done, tf.int64))

        self.epochs_done.assign_add(1)

        log = {'loss': loss, 'acc@1': acc_1, 'acc@5': acc_5}

        log = tf.nest.map_structure(tf.TensorArray.stack, log)

        log = tf.nest.map_structure(tf.reduce_mean, log)

        return log

    def train(self, datasets: List[tf.data.Dataset], args: Args):
        logs = []

        train_ds, val_ds = datasets
            # TODO: what should be loaded e.g. for transfer?

        start_epoch = self.epochs_done.numpy()

        for epoch_idx in range(start_epoch, args.epochs):
            train_result = self.train_fn(train_ds, args.log_every)
            val_result, improved = self.evaluate(val_ds)

            self.save_weights(
                os.path.join(args.ckpt_path, 'last')
            )

            train_result, val_result, improved = tf.nest.map_structure(lambda x: x.numpy(),
                                                                       [train_result, val_result, improved])

            logs.append(
                dict(train_result, **{f'val_{metric}': value for metric, value in val_result.items()})
            )

            met_strs = map('{}: {:.5f}'.format, *zip(*logs[-1].items()))

            print(f'Epoch {epoch_idx} | ' + ', '.join(met_strs))

            if improved:
                print('acc@1 improved, saving weights')
                self.save_weights(
                    os.path.join(args.ckpt_path, 'best')
                )

        return logs


def get_model(args: Args):
    if args.lr_schedule == 'cosine':
        lr = tf.keras.experimental.CosineDecay(
            initial_learning_rate=args.base_lr,
            decay_steps=args.warmup_steps,
        )
    elif args.lr_schedule == 'step':
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.base_lr,
            decay_steps=args.warmup_steps,
            end_learning_rate=args.end_lr
        )
    else:
        lr = args.base_lr

    # I'm not sure what the value they give refers to but I am using it
    # as the argument to `weight_decay`
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, weight_decay=args.weight_decay
    )

    writers = {name: tf.summary.create_file_writer(args.tbd_path + f'/{name}')
               for name in ['train', 'val', 'lr']}

    assert (args.image_size % args.patch_size) == 0
    model = ViT(
        optimizer=optimizer,
        writers=writers,
        args=args
    )

    if args.resume:
        model.load_weights(args.resume_path)

    return model


def get_dataset(args: Args):

    if args.dataset_name == 'cifar10':
        if os.path.isdir(args.data_dir):

            train_ds = tfds.load("cifar10", split="train",
                                 decoders={'image': tfds.decode.SkipDecoding()},
                                 shuffle_files=True,
                                 data_dir=args.data_dir)
            val_ds = tfds.load("cifar10", split="test",
                               decoders={'image': tfds.decode.SkipDecoding()},
                               shuffle_files=False,
                               data_dir=args.data_dir)
        else:
            with open(args.data_dir, 'rb') as f:
                data = pickle.load(f)

            train_data = data['train']
            test_data = data['test']

            train_ds = tf.data.Dataset.from_tensor_slices(train_data)
            val_ds = tf.data.Dataset.from_tensor_slices(test_data)

    elif args.dataset_name == 'resisc45':

        # data = pickle.load(tf.io.gfile.GFile(args.data_dir, 'rb'))
        #
        # train_ds = tf.data.Dataset.from_tensor_slices(
        #     data['train']
        # )
        # val_ds = tf.data.Dataset.from_tensor_slices(
        #     data['val']
        # )
        # > For Resisc45, we use only 60% of the training split for training, and
        # another 20% for validation, and 20% for computing test metrics
        train_ds, val_ds, _ = tfds.load(
            'resisc45',
            split=['train[:60%]', 'train[60%:80%]', 'train[80%:]'],
            data_dir=args.data_dir,
            decoders={'image': tfds.decode.SkipDecoding()}
        )

    else:
        raise ValueError("Only cifar and resisc45 are supported for now")

    def _transf(x, mode='train'):
        img = x['image']
        if args.dataset_name == 'resisc45':
            #img = tf.io.read_file(img)
            img = tf.io.decode_jpeg(img)
        if mode == 'train':
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(img),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,  # Don't enforce a minimum area.
                use_image_if_no_bounding_boxes=True)
            img = tf.slice(img, begin, size)

        img = tf.image.resize(img, [args.image_size, args.image_size])
        img.set_shape([args.image_size, args.image_size, 3])

        if mode == 'train' :
            img = tf.image.random_flip_left_right(img)
            if args.randaug_num_layers > 0 and args.randaug_magnitude > 0:
                # randaugment requires uint8
                img = tf.cast(img, tf.uint8)
                img = distort_image_with_randaugment(img, num_layers=args.randaug_num_layers,
                                                     magnitude=args.randaug_magnitude)
                img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        return dict(image=img, label=x['label'],
                    label_onehot=tf.one_hot(x['label'], depth=args.num_classes))

    train_ds = train_ds.shuffle(len(train_ds)).map(
        partial(_transf, mode='train')
        ,  tf.data.experimental.AUTOTUNE)

    if args.debug_train:
        train_ds = train_ds.take(args.debug_train_steps * args.batch_size)

    train_ds = train_ds.batch(args.batch_size)

    if args.debug_val:
        val_ds = val_ds.take(args.debug_val_steps * args.batch_size)

    val_ds = val_ds.map(
    partial(_transf, mode='val')
        ,  tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size)

    return train_ds.prefetch(tf.data.experimental.AUTOTUNE), val_ds.prefetch(tf.data.experimental.AUTOTUNE)


class Trainer:
    def __init__(self, args: Args):
        self.args = args
        os.makedirs(args.model_path, exist_ok=False)
        os.makedirs(args.ckpt_path, exist_ok=False)
        os.makedirs(args.tbd_path, exist_ok=False)

        with open(os.path.join(args.model_path, 'args.json'), 'w') as f:
            json.dump(fp=f, obj=attr.asdict(args))

        self.model = get_model(args)
        self.train_ds, self.val_ds = get_dataset(args)

    def train(self):
        self.model.train([self.train_ds, self.val_ds], self.args)


if __name__ == '__main__':
    cfg = Args(save_path='./tmp', num_classes=10,
               batch_size=4,
               base_lr=0.001,
               weight_decay=0.,
               mixup_alpha=0.1,
               last_res_prob=0.5,
               randaug_magnitude=5,
               randaug_num_layers=2,
               data_dir='/Volumes/GoogleDrive/My Drive/transformer/data/cifar10/arrays.pickle',
               debug_train=True, debug_val=True,
               debug_train_steps=8,
               debug_val_steps=4,
               patch_size=16,
               log_every=1,
               epochs=2)
    trainer = Trainer(cfg)
    trainer.train()












