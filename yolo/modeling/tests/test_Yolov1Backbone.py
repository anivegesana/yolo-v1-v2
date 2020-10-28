import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import functools
import os
import unittest

from yolo.modeling.backbones import backbone_builder as builder


@parameterized.named_parameters(
    ("simple", "yolov1_backbone", (1, 448, 448, 3)),
)
class Yolov1BackboneTest(tf.test.TestCase, parameterized.TestCase):

    def test_pass_through(self, model, input_shape):
        check = {
            '64': [input_shape[0], input_shape[1] // 4, input_shape[2] // 4, 64],
            '192': [input_shape[0], input_shape[1] // 8, input_shape[2] // 8, 192],
            '512': [input_shape[0], input_shape[1] // 16, input_shape[2] // 16, 512],
            '1024_1': [input_shape[0], input_shape[1] // 32, input_shape[2] // 32, 1024],
            '1024_2': [input_shape[0], input_shape[1] // 32, input_shape[2] // 32, 1024]
        }

        init = tf.random_normal_initializer()
        x = tf.Variable(initial_value=init(shape=input_shape, dtype=tf.float32))
        y = builder.Backbone_Builder(model)(x)

        y_shape = {key: value.shape.as_list() for key, value in y.items()}
        self.assertAllEqual(check, y_shape)
        print(y_shape, check)

    def test_gradient_pass_though(self, model, input_shape):
        check = {
            '64': [input_shape[0], input_shape[1] // 4, input_shape[2] // 4, 64],
            '192': [input_shape[0], input_shape[1] // 8, input_shape[2] // 8, 192],
            '512': [input_shape[0], input_shape[1] // 16, input_shape[2] // 16, 512],
            '1024_1': [input_shape[0], input_shape[1] // 32, input_shape[2] // 32, 1024],
            '1024_2': [input_shape[0], input_shape[1] // 32, input_shape[2] // 32, 1024]
        }

        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = builder.Backbone_Builder(model)

        init = tf.random_normal_initializer()
        x = tf.Variable(initial_value=init(shape=input_shape, dtype=tf.float32))
        y = {
            key: tf.Variable(initial_value=init(shape=value, dtype=tf.float32))
            for key, value in check.items()
        }

        with tf.GradientTape() as tape:
            x_hat = test_layer(x)

            losses = 0
            for key in y:
                grad_loss = loss(x_hat[key], y[key])
                losses += grad_loss

        grad = tape.gradient(losses, test_layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

        self.assertNotIn(None, grad)


if __name__ == "__main__":
    tf.test.main()
