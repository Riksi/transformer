import tensorflow as tf

def main(config):
    model = get_model(config)
    train_dataset = get_dataset(config, mode='train')
    val_dataset = get_dataset(config, mode='val')

    for epoch in range(config.epochs):
        for itr, (inputs, targets) in enumerate(train_dataset):
            preds, loss = model.train_step(inputs, targets)
            # TODO: Do something with loss

        for val_itr, (inputs, targets) in enumerate(val_dataset):
            preds, loss = model.valid_step(inputs, targets)
            # TODO: Do something with loss






