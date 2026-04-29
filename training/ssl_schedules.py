
from __future__ import annotations

from typing import Callable

import tensorflow as tf


TemperatureInput = float | int | tf.keras.optimizers.schedules.LearningRateSchedule | Callable[[tf.Tensor], tf.Tensor]
LrInput = float | int | tf.keras.optimizers.schedules.LearningRateSchedule


class LinearWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_value: float, end_value: float, warmup_steps: int):
        super().__init__()
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.warmup_steps = int(max(warmup_steps, 1))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        ratio = tf.minimum(step / float(self.warmup_steps), 1.0)
        return self.start_value + ratio * (self.end_value - self.start_value)

    def get_config(self):
        return {
            'start_value': self.start_value,
            'end_value': self.end_value,
            'warmup_steps': self.warmup_steps,
        }


def resolve_schedule_value(schedule: TemperatureInput, step: tf.Tensor) -> tf.Tensor:
    if isinstance(schedule, (float, int)):
        return tf.cast(float(schedule), tf.float32)
    value = schedule(step)
    return tf.cast(value, tf.float32)
