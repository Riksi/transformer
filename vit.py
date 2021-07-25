import tensorflow as tf
from easydict import EasyDict
from typing import List, Union, Callable

from embedding import PositionalEncoding, ScaledEmbedding
from blocks import MultiHeadAttention, FeedForward
import os
import attr
import datetime
import tensorflow_datasets as tfds

from functools import partial

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
# - [ ] Add dataloader for CIFAR-10 and get it running
# - [x] CIFAR-10 bounding box
# - [x] Log lr
# -------------------------------------------------------------------------------------
# - [ ] I wonder how the encodings are interpolated
#   - DO NOT TRY THIS FOR THE UNTIL YOU HAVE TRAINED A MODEL AS IT WILL BE A HUGE TIME SINK

@attr.s(auto_attribs=True)
class Args(object):
    save_path: str
    batch_size: int
    num_classes: int
    grad_clip: Union[float, None] = None
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
    warmup_steps: int = 10000
    base_lr: float = 3e-3
    end_lr: float = 1e-5
    weight_decay: float = 0.3
    dropout: float = 0.1
    lr_schedule: Union[str, None] = 'cosine'
    resume: bool = False
    resume_path: str = None
    model_name: str = attr.ib(
        default=attr.Factory(
            lambda self: f"{self.arch}_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}",
            takes_self=True
        )
    )
    model_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.save_path, self.model_name)
        )
    )
    ckpt_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.model_path, 'ckpts')
        )
    )
    tbd_path: str = attr.ib(
        default=attr.Factory(
            lambda self: os.path.join(self.model_path, 'tbd')
        )
    )



@tf.function
def get_acc(logits, labels, in_top=(1,)):
    # logits: [B, Z]
    # labels: [B]
    # [B, Z] -> [B, T]
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
    # [B, A]
    contains = tf.reduce_any(tf.equal(top, labels[:, None, None]), axis=-1)
    # [A]
    value = tf.reduce_mean(tf.cast(contains, tf.float32), axis=0)

    for idx, i in enumerate(in_top):
        res[f'acc@{i}'] = value[idx]

    return res

class MLP(FeedForward):
    def __init__(self, *args, dropout=0.0, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        if dropout > 0.0:
            self.use_dropout = True
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False

    def call(self, x, training=None):
        x = self.dense1(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        return self.dense2(x)


class ViTBlock(tf.keras.models.Model):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.0):
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.attn = MultiHeadAttention(model_dim, num_heads=num_heads)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        # ODOT: Confirm this is the structure of MLP - confirmed
        # (difference from before was gelu instead of relu)
        self.mlp = MLP(hidden_dim=ff_dim, output_dim=model_dim, activation=tf.nn.gelu, dropout=dropout)
        if dropout > 0.:
            self.use_dropout = True
            self.dropout_layer1 = tf.keras.layers.Dropout(dropout)
            self.dropout_layer2 = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False

    def call(self, inputs, training=None, **kwargs):
        skip = inputs
        # TODO: are q,k,v all passed here - I would think so
        x = self.layer_norm_1(inputs)
        # TODO: any mask here? I don't think there would be as not autoregressive
        x, attn = self.attn(x, x, x)
        if self.use_dropout:
            x = self.dropout_layer1(x)
        x = skip + x

        skip = x
        x = self.mlp(self.layer_norm_2(x))
        if self.use_dropout:
            x = self.dropout_layer2(x)
        x = skip + x

        return x, attn


class Encoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0):
        super(Encoder, self).__init__()
        self.blocks = [
            ViTBlock(dim, ff_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]

    def call(self, inputs, mask=None, training=True):
        attn_maps = []
        query = inputs
        for block in self.blocks:
            query, attn = block(
                          query=inputs, key=inputs, value=inputs,
                          mask=mask, training=training)
            attn_maps.append(attn)
        return query, attn_maps


class ImageEmbedding(tf.keras.models.Model):
    def __init__(self, patch_size, model_dim, dropout=0.):
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(units=self.model_dim, use_bias=False)
        if dropout > 0.:
            self.use_dropout = True
            self.dropout_layer = tf.keras.layers.Dropout(dropout)
        else:
            self.use_dropout = False
        self.cls_token = tf.Variable(tf.zeros([self.d_model]))

    def call(self, inputs, pos_encoding, training=None):
        # inputs: [B, H, W, F]
        B, H, W, F = tf.shape(inputs)
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
                 patch_size,
                 num_classes,
                 writers,
                 optimizer,
                 grad_clip=None,
                 encoding_size=(224 // 16)**2,
                 model_dim=768,
                 num_heads=12,
                 dropout=0.1,
                 ff_dim=3072,
                 num_layers=12,
                 fine_tune=False):
        super(ViT, self).__init__()
        self.model_dim = model_dim
        self.input_embedding = ImageEmbedding(patch_size, model_dim)
        self.pos_encoding = tf.Variable(shape=[encoding_size, model_dim])
        self.grad_clip = grad_clip
        self.encoder = Encoder(dim=model_dim,  # 256
                               ff_dim=ff_dim,  # 2048
                               num_heads=num_heads,  # 8
                               dropout=dropout,
                               num_blocks=num_layers)

        self.layer_norm = tf.keras.layers.LayerNormalization()
        # > The classification head is implemented by a MLP
        # with one hidden layer at pre-training
        # time and by a single linear layer at fine-tuning time.
        if not fine_tune:
            self.mlp_head = tf.keras.models.Sequential(
                tf.keras.layers.Dense(units=model_dim, activation='tanh'),
                tf.keras.layers.Dense(units=num_classes,
                                      kernel_initializer=tf.initializers.zeros)
            )
        else:
            self.mlp_head = tf.keras.layers.Dense(units=num_classes,
                                                  kernel_initializer=tf.initializers.zeros)

        self.optimizer = optimizer
        self.writers = writers
        self.epochs_done = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.best_epoch = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.best_acc = tf.Variable(0., dtype=tf.float32, trainable=False)

    def call(self, x, mask=None, training=True):
        # x: [B, H, W, F]
        # [T, D]
        pe = self.pos_encoding[:tf.shape(x)[1]]
        # [B, T=H*W/P^2, D=P^2*F]
        x = self.input_embedding(x) + pe
        # [B, T, D]
        x, attn = self.encoder(x, mask=mask, training=training)
        x = self.layer_norm(x)
        # [B, D]
        cls_feature = x[:, 0]
        # [B, Z]
        logits = self.mlp_head(cls_feature)
        return logits, x, attn

    def get_losses_for(self, data):
        logits, x, attn = self.call(data['image'], data.mask, training=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=data.targets)
        )
        acc = get_acc(logits, data['label'], (1, 5)
        return dict(
            loss=loss, acc=acc
        )

    def train_step(self, data):
        with tf.GradientTape() as tape:
            result = self.get_losses_for(data)
        grads = tape.gradient(result['loss'], self.trainable_variables)
        if self.grad_clip is not None:
            grads, gnorm = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return result

    def evaluate(self, dataset):
        log = {'loss': tf.constant(0.),
               'acc@1': tf.constant(0.),
               'acc@5': tf.constant(0.)}
        count = 0
        for idx, batch in enumerate(dataset):
            result = self.get_losses_for(batch)
            count += len(batch)
            for metric, value in result.items():
                log[metric] += (result * len(batch))

        log =  {metric: value / count for metric, value in log.items()}

        improved = tf.constant(False)
        if log['acc@1'] > self.best_acc:
            improved = tf.constant(True)
            self.best_acc.assign(log['acc@1'])
            self.best_epoch.assign(self.epochs_done)

        epoch_idx = self.epochs_done - 1
        for metric, value in log.items():
            with self.summary_writers['val'].as_default():
                tf.summary.scalar(name=metric, data=value, step=tf.cast(epoch_idx, tf.int64))

        return log, improved


    @tf.function
    def train_fn(self, dataset, log_every):
        steps_per_epoch = tf.cast(len(dataset), tf.int32)
        log_steps = steps_per_epoch // log_every + 1

        loss = tf.TensorArray(size=log_steps, dtype=tf.float32)
        acc_1 = tf.TensorArray(size=log_steps, dtype=tf.float32)
        acc_5 = tf.TensorArray(size=log_steps, dtype=tf.float32)

        log_idx = 0

        global_itr = self.epochs_done * steps_per_epoch - 1

        for idx, inputs in enumerate(dataset):

            global_itr += 1

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
                    with self.summary_writers['train'].as_default():
                        with tf.name_scope(''):
                            tf.summary.scalar(name=metric, data=value, step=tf.cast(global_itr, tf.int64))

        with self.summary_writers['lr'].as_default():
            if isinstance(self.optimizer.lr, Callable):
                lr = self.optimizer.lr(global_itr)
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

        if args.resume:
            self.load_weights(args.resume_path)

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

    optimizer = tf.optimizers.Adam(
       learning_rate=lr, beta_1=0.9, beta_2=0.999
    )
    writers = {name: tf.summary.create_file_writer(args.tbd_path + f'/{name}')
               for name in ['train', 'val', 'lr']}

    assert (args.image_size % args.patch_size) == 0
    model = ViT(
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        optimizer=optimizer,
        writers=dict(writers, **writers),
        grad_clip=args.grad_clip,
        encoding_size=(args.image_size // args.patch_size) ** 2,
        model_dim=768,
        num_heads=12,
        dropout=0.1,
        ff_dim=3072,
        num_layers=12,
        fine_tune=False
    )

    return model

def get_dataset(args: Args):

    @tf.function
    def _transf(x, mode='train'):
        img = x['image']
        if mode == 'train':
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(img),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,  # Don't enforce a minimum area.
                use_image_if_no_bounding_boxes=True)
            img = tf.slice(img, begin, size)
        img = tf.image.resize(img, [args.image_size, args.image_size])
        if mode=='train':
            img = tf.image.random_flip_left_right(img)
        img = (img - 127.5) / 127.5
        x['image'] = img
        return x

    if args.debug_train:
        train_split = f'train[:{args.debug_train_steps * args.batch_size}]'
    else:
        train_split = 'train'

    if args.debug_val:
        val_split = f'test[:{args.debug_val_steps * args.batch_size}]'
    else:
        val_split = 'test'

    train_ds, val_ds = tfds.load('cifar10',
                        split=[train_split, val_split],
                        data_dir=args.data_dir)
    train_ds = train_ds.shuffle().map(_transf, tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(args.batch_size)

    val_ds = val_ds.map(
        partial(_transf, mode='val'), tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.batch(args.batch_size)


    return train_ds.prefetch(tf.data.experimental.AUTOTUNE), val_ds.prefetch(tf.data.experimental.AUTOTUNE)


def run(args):
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.ckpt_path, exist_ok=False)
    os.makedirs(args.tbd_path, exist_ok=False)

    model = get_model(args)
    train_ds, val_ds = get_dataset(args)

    model.train([train_ds, val_ds], args)


if __name__ == '__main__':
    cfg = Args(save_path='./tmp', num_classes=10,
               batch_size=4, data_dir='./data',
               debug_train=True, debug_val=True,
               patch_size=8,
               epochs=2)
    run(cfg)
    # args = EasyDict()
    # args.patch_size = 16
    # args.epochs = 300
    # args.warmup_steps = 100000
    # args.base_lr = 3e-3
    # args.end_lr = 1e-5
    # args.weight_decay = 0.3
    # args.dropout = 0.1
    # args.lr_schedule = 'cosine'
    # args.tbd_path = 'tmp'












