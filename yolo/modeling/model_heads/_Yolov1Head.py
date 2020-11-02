import tensorflow as tf
import tensorflow.keras as ks

import importlib
import collections
from typing import *

import yolo.modeling.building_blocks as nn_blocks
from . import configs

from yolo.modeling.backbones.backbone_builder import Backbone_Builder

class Yolov1Head(tf.keras.Model):
    def __init__(self,
                 model="regular",
                 classes=20,
                 S=7,
                 boxes=3,
                 config=None,
                 input_shape=(None, None, None, 512),
                 **kwargs):
        """
        Args:
            model: to generate a standard yolo head, we have string key words that can accomplish this
                    regular -> corresponds to yolov1
            S: integer for the number of grid cells in SxS output
            classes: integer for the number of classes in the prediction
            boxes: integer for the total number of bounding boxes
            config:

        """

        self._config = config
        self._classes = classes
        self._S = S
        self._boxes = boxes
        self._model_name = model

        if not isinstance(config, Dict):
            self._config = self.load_cfg(model)
        else:
            self._model_name = "custom_head"

        self._input_shape = input_shape
        self._output_depth = (boxes * 5) + classes

        inputs = ks.layers.Input(shape=self._input_shape[None][1:])
        outputs = self._connect_layers(self._config, inputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=self._model_name,
                         **kwargs)
        return

    @classmethod
    def load_cfg(self, model):
        """find the config file and load it for use"""
        try:
            head = importlib.import_module('.yolov1_'+ model,
                                            package=configs.__package__).head
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.yolov1_' + model:
                raise ValueError(f"Invlid head '{model}'") from e
            else:
                raise
        return head_build_block_specs(head)

    def _connect_layers(self, layers, inputs):
        x = inputs
        for layer in layers:
            if layer.name == "DarkConv":
                x = nn_blocks.DarkConv(filters=layer.filters,
                                        kernel_size=layer.kernel_size,
                                        strides=layer.strides,
                                        padding=layer.padding,
                                        activation=layer.activation)(x)
            elif layer.name == "Local":
                if layer.activation == "leaky":
                    act = ks.layers.LeakyReLU(alpha=0.1)
                else:
                    act = layer.activation

                if layer.padding == "same" and layer.kernel_size != 1:
                    padding = layer.kernel_size // 2
                    x = ks.layers.ZeroPadding2D(
                        ((padding, padding), (padding, padding))  # symetric padding
                    )(x)
                x = ks.layers.LocallyConnected2D(filters=layer.filters,
                                                 kernel_size=layer.kernel_size,
                                                 strides=layer.strides,
                                                 padding="valid",
                                                 activation=act)(x)
            elif layer.name == "Dropout":
                x = ks.layers.Dropout(rate=layer.filters)(x)
            elif layer.name == "Connected":
                x = ks.layers.Dense(units=layer.filters,
                                    activation=layer.activation)(x)
        #print(self._S, self._S, self._output_depth)
        return x #ks.layers.Reshape((self._S, self._S, self._output_depth))(x)

class HeadBlockConfig(object):
    def __init__(self, layer, filters, kernel_size,
                 strides, padding, activation):
        '''
        get layer config to make code more readable

        Args:
            layer: string layer name
            filters: integer for the filter for this layer, or the output depth
            kernel_size: integer or none, if none, it implies that the the building
                         block handles this automatically. not a layer input
            padding:
            activation:
        '''
        self.name = layer
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        return

    def __repr__(self):
        return f"layer: {self.name}, filters: {self.filters}, kernel_size: {self.kernel_size}, strides: {self.strides}, padding: {self.padding}, activation: {self.activation}\n"

def head_build_block_specs(config):
        specs = []
        for layer in config:
            specs.append(HeadBlockConfig(*layer))
        return specs
