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


class OrthoAlphaScheduleCallback(keras.callbacks.Callback):
    """Curriculum schedule for ortho_alpha_var: 0 → target_alpha.

    Phase 1 — Hold (0 → hold_steps):
        ortho_alpha = 0.0  (pure entropy loss, no ortho pressure)

    Phase 2 — Warmup (hold_steps → hold_steps + warmup_steps):
        ortho_alpha linearly increases 0.0 → target_alpha

    Phase 3 — Hold at target (hold_steps + warmup_steps → end):
        ortho_alpha = target_alpha

    Rationale
    ---------
    Entropy loss first stabilises head diversity (prevents early collapse).
    Ortho loss then refines head orthogonality once the distribution is
    healthy. Curriculum ordering mirrors DINO teacher temperature warmup.

    Works with any model that exposes an ``ortho_alpha_var`` tf.Variable.

    Usage
    -----
    step_size   = 1858 // batch_size          # steps per epoch
    hold_steps  = step_size * 20              # pure entropy for 20 epochs
    ramp_steps  = step_size * 30              # ramp to target over 30 epochs

    cb = OrthoAlphaScheduleCallback(
        target_alpha=0.5,
        hold_steps=hold_steps,
        warmup_steps=ramp_steps,
        verbose=True,
    )
    model.fit(..., callbacks=[cb])
    """

    def __init__(
        self,
        target_alpha: float = 0.5,
        hold_steps: int = 0,
        warmup_steps: int = 1000,
        verbose: bool = False,
    ):
        super().__init__()
        self.target_alpha = float(target_alpha)
        self.hold_steps   = max(int(hold_steps), 0)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.verbose      = verbose
        self._global_batch = 0

    def on_train_begin(self, logs=None):
        self._global_batch = 0
        self._set_alpha(0.0)  # always start at 0 (pure entropy)
        if self.verbose:
            print(
                f"\n[OrthoAlphaSchedule] init  ortho_alpha = 0.0  "
                f"(hold={self.hold_steps} steps, "
                f"warmup={self.warmup_steps} steps, "
                f"target={self.target_alpha})"
            )

    def on_train_batch_end(self, batch, logs=None):
        self._global_batch += 1
        g = self._global_batch

        if g <= self.hold_steps:
            new_alpha = 0.0
        else:
            prog = min((g - self.hold_steps) / self.warmup_steps, 1.0)
            new_alpha = prog * self.target_alpha

        self._set_alpha(new_alpha)

        if self.verbose and g % 20 == 0:
            phase = (
                "hold" if g <= self.hold_steps
                else "ramp" if g <= self.hold_steps + self.warmup_steps
                else "stable"
            )
            print(
                f"\n[OrthoAlphaSchedule] step={g:5d}  "
                f"phase={phase:6s}  ortho_alpha={new_alpha:.4f}"
            )

    def _set_alpha(self, value: float):
        if hasattr(self.model, 'ortho_alpha_var'):
            self.model.ortho_alpha_var.assign(float(value))
        else:
            import warnings
            warnings.warn(
                'OrthoAlphaScheduleCallback: model has no ortho_alpha_var attribute. '
                'Callback has no effect.',
                stacklevel=2,
            )
