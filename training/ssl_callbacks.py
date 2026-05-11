from __future__ import annotations

import keras
import tensorflow as tf


class TeacherTempWarmupCallback(keras.callbacks.Callback):
    """Linearly warm-up teacher_temp_var from start_value → end_value
    over warmup_steps *batches*, then hold at end_value.

    Works with any model that exposes a ``teacher_temp_var`` tf.Variable.
    Updated at on_train_batch_end so the next batch sees the new value.

    Usage
    -----
    cb = TeacherTempWarmupCallback(
        start_value=0.04,
        end_value=0.10,
        warmup_steps=1160,   # e.g. 10 epochs * 116 steps/epoch
        verbose=True,
    )
    model.fit(..., callbacks=[cb])
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        warmup_steps: int,
        verbose: bool = False,
    ):
        super().__init__()
        self.start_value  = float(start_value)
        self.end_value    = float(end_value)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.verbose      = verbose
        self._global_batch = 0

    def on_train_begin(self, logs=None):
        """Reset counter and set initial value at the start of fit()."""
        self._global_batch = 0
        self._set_temp(self.start_value)
        if self.verbose:
            print(f"\n[TeacherTempWarmup] init  teacher_temp = {self.start_value:.6f}")

    def on_train_batch_end(self, batch, logs=None):
        self._global_batch += 1
        ratio = min(self._global_batch / self.warmup_steps, 1.0)
        new_temp = self.start_value + ratio * (self.end_value - self.start_value)
        self._set_temp(new_temp)

        if self.verbose and self._global_batch % 20 == 0:
            print(
                f"\n[TeacherTempWarmup] step={self._global_batch:5d}  "
                f"ratio={ratio:.4f}  teacher_temp={new_temp:.6f}"
            )

    def _set_temp(self, value: float):
        if hasattr(self.model, "teacher_temp_var"):
            self.model.teacher_temp_var.assign(float(value))
        else:
            import warnings
            warnings.warn(
                "TeacherTempWarmupCallback: model has no teacher_temp_var attribute. "
                "Callback has no effect.",
                stacklevel=2,
            )


class LambdaSimmIMWarmupCallback(keras.callbacks.Callback):
    """Warm up model.lambda_simmim over training batches.

    Schedule:
      - hold start_value until start_step
      - linearly increase to end_value over warmup_steps
      - hold end_value afterwards

    Works with any model exposing a writable ``lambda_simmim`` attribute.
    """

    def __init__(
        self,
        start_value: float = 0.0,
        end_value: float = 0.3,
        start_step: int = 0,
        warmup_steps: int = 1000,
        verbose: bool = False,
    ):
        super().__init__()
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        self.start_step = max(int(start_step), 0)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.verbose = verbose
        self._global_batch = 0

    def on_train_begin(self, logs=None):
        self._global_batch = 0
        self._set_lambda(self.start_value)
        if self.verbose:
            print(f"\n[LambdaSimmIMWarmup] init lambda_simmim = {self.start_value:.6f}")

    def on_train_batch_end(self, batch, logs=None):
        self._global_batch += 1
        if self._global_batch <= self.start_step:
            new_value = self.start_value
        else:
            prog = min((self._global_batch - self.start_step) / self.warmup_steps, 1.0)
            new_value = self.start_value + prog * (self.end_value - self.start_value)
        self._set_lambda(new_value)
        if self.verbose and self._global_batch % 20 == 0:
            print(
                f"\n[LambdaSimmIMWarmup] step={self._global_batch:5d} "
                f"lambda_simmim={new_value:.6f}"
            )

    def _set_lambda(self, value: float):
        if hasattr(self.model, 'lambda_simmim'):
            self.model.lambda_simmim = float(value)
        else:
            import warnings
            warnings.warn(
                'LambdaSimmIMWarmupCallback: model has no lambda_simmim attribute. '
                'Callback has no effect.',
                stacklevel=2,
            )
