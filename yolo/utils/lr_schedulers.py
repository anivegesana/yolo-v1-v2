import tensorflow as tf
import tensorflow.keras as ks
import numpy as np


class CosineDecay(ks.callbacks.LearningRateScheduler):
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps,
                 hold_steps, end_learning_rate=0.01, offset=0.00, verbose=0):
        self._warmup_steps = warmup_steps
        self._initial_learning_rate = initial_learning_rate
        self._learning_rate = initial_learning_rate
        self._end_learning_rate = end_learning_rate
        self._decay_steps = decay_steps
        self._offset = offset
        self._hold_steps = hold_steps
        super().__init__(self._schedule, verbose=verbose)

    def _schedule(self, epoch, lr):
        if epoch == 0:
            lr = 0.0  # apparently starts as 0.01 and not 0.0
        if epoch < self._warmup_steps:
            lr += self._initial_learning_rate / self._warmup_steps
        elif lr >= self._offset:
            lr = 0.5 * self._initial_learning_rate * (
                1 + np.cos(
                    np.pi * (epoch - self._warmup_steps - self._hold_steps) /
                    (self._decay_steps - self._warmup_steps - self._hold_steps)
                )
            )
        else:
            lr = self._offset
        return lr


class YOLOv1StepDecay(ks.callbacks.LearningRateScheduler):
    """
    Described in YOLO v1 paper
        - Start with 10^-3
        - Increase to 10^-2 over 'first few' epochs
        - Train with 10^-2 for 75 epochs
        - Train with 10^-3 for 30 epochs
        - Train with 10^-4 for 30 epochs
    Total ~150 epochs.
    """
    def __init__(self, verbose=0):
        super().__init__(self._schedule, verbose=verbose)

    def _schedule(self, epoch, lr):
        if epoch < 15:
            return 10e-3 + (10e-2 - 10e-3) / 15 * epoch
        elif epoch < 90:
            return 10e-2
        elif epoch < 120:
            return 10e-3
        else:
            return 10e-4


if __name__ == "__main__":
    model = ks.Sequential(layers=[ks.layers.Dense(10)])
    x = np.ones(10)
    y = np.ones(10)
    learning_rate = CosineDecay(0.1, 5, 150, 0, verbose=1)
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=['accuracy']
    )
    model.fit(x, y, epochs=150, callbacks=[learning_rate])
