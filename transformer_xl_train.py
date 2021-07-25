import tensorflow as tf
import json
import math

# TODO:
# - [ ] Fix all the functions here
# - [ ] Implement data loading
# - [ ] Modify Transformer


def get_model(args, n_token, cutoffs):

    # @tf.function
    # def model_fn(inp, tgt, mems, is_training):
        # inp = tf.transpose(inp, [1, 0])
        # tgt = tf.transpose(inp, [1, 0])

    if args.init == "uniform":
        initialiser = tf.initializers.random_uniform(
            minval=args.init_range,
            maxval=args.init_range,
            seed=None
        )
    elif args.init == "normal":
        initialiser = tf.initializers.random_normal(
            stddev=args.init_std,
            seed=None
        )
        proj_initialiser = tf.initializers.random_normal(
            stddev=args.proj_init_std,
            seed=None
        )

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if args.proj_share_all_but_first:
        for i in range(1, len(tie_projs)):
            tie_projs[i] = True

    model = transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initialiser,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=False,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    num_params = sum([np.prod(v.shape) for v in model.trainable_variables])

    print("#params: {}".format(num_params))

    return model


def train(args, dataset, model: tf.keras.models.Model, optim: tf.keras.optimizers.Optimizer):
    prev_itr = -1.

    ckpt_path = f"{args.model_dir}/model.ckpt"
    ckpt = tf.train.Checkpoint(
        transformer=model,
        optimizer=optim
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, ckpt_path, max_to_keep=1
    )

    config_path = f"{args.model_dir}/config.json"
    # TODO: maybe YAML?
    with open(config_path) as f:
        json.dump(fp=config_path, obj=dict(args, ckpt_path=ckpt_path))

    #     if ckpt_manager.latest_checkpoint:
    #         ckpt.restore(ckpt_manager.latest_checkpoint)
    #         print('Latest checkpoint restored')

    mems = [tf.zeros([args.mem_len, args.batch_size, args.d_model])
            for _ in range(args.n_layer)]

    for itr, (inputs, targets) in enumerate(dataset):
        total_loss, new_mems, gnorm = model.train_step(inputs=inputs, targets=targets, mems=mems)
        mems.append(new_mems)

        if itr > 0 and itr % args.iterations == 0:
            loss = total_loss / (itr - prev_itr)

            printstr = f"[{itr}] | gnorm {gnorm:.2f} lr {optim.lr.numpy():8.6f}"
            printstr += f" | loss {loss:.2f}"
            printstr += f" | pplx {math.exp(loss):>7.2f}, bpc {loss / math.log(2):>7.4f}"

            print(printstr)

        if itr > 0 and itr % args.save_steps == 0:
            ckpt_manager.save()
            print(f"Model saved in {ckpt_path}")

        if itr == args.train_steps:
            break


# TODO: Complete this
def evaluate(args, dataset, model: tf.keras.models.Model, optim: tf.keras.optimizers.Optimizer):
    num_batch = args.batch_size
    if args.max_eval_batch > 0:
        num_batch = args.max_eval_batch

    print(f"num of batches {num_batch}")

    mems = [tf.zeros([args.mem_len, args.batch_size, args.d_model])
            for _ in range(args.n_layer)]

    total_loss, total_count = 0, 0
    for itr, (inputs, targets) in enumerate(dataset):
        loss, new_mems, count = model.valid_step(inputs=inputs, targets=targets, mems=mems)
        total_loss += loss
        total_count += count
        mems.append(new_mems)

    avg_loss = total_loss / total_count

    print("loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


# Might be easier to do after setting up data
def main(unused_argv):
      del unused_argv  # Unused

      #tf.logging.set_verbosity(tf.logging.INFO)

      # Get corpus info
      corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
      n_token = corpus_info["vocab_size"]
      cutoffs = corpus_info["cutoffs"][1:-1]
      tf.logging.info("n_token {}".format(n_token))

      if FLAGS.do_train:
        train(n_token, cutoffs, "/gpu:0")
      if FLAGS.do_eval:
        evaluate(n_token, cutoffs, "/gpu:0")