import tensorflow as tf


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps):
        self._d_model = d_model
        self._warmup_steps = warmup_steps
        self.d_model_term = d_model ** (-0.5)
        self.warmup_term = warmup_steps ** (-1.5)

    def get_config(self):
        return dict(d_model=self._d_model, warmup_steps=self.warmup_steps)

    def __call__(self, step):
        step_term = tf.math.minimum(step ** (-0.5), step * self.warmup_term)
        return self.d_model_term * step_term