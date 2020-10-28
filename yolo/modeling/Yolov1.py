import tensorflow as tf
import tensorflow.keras as ks
from typing import *

import yolo.modeling.base_model as base_model
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov1Head import Yolov1Head
from yolo.modeling.building_blocks import YoloLayer

from yolo.utils.file_manager import download
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter, load_weights_dnBackbone, load_weights_dnHead

class Yolov1(base_model.Yolo):
    def __init__(
            self,
            input_shape=[None, None, None, 3],
            model="regular",  # options {regular, spp, tiny}
            classes=20,
            backbone=None,
            head=None,
            boxes=3,
            weight_decay = 5e-4,
            policy="float32",
            using_rt=False,
            **kwargs):
        super().__init__(**kwargs)

        #required inputs
        self._input_shape = input_shape
        self._classes = classes
        self._type = model
        self._boxes = boxes
        self._built = False
        self._custom_aspects = False

        #setting the running policy
        if type(policy) != str:
            policy = policy.name
        self._og_policy = policy
        self._policy = tf.keras.mixed_precision.experimental.global_policy(
        ).name
        self.set_policy(policy=policy)

        #init models
        self.model_name = model
        self._model_name = None
        self._backbone_name = None
        self._backbone_cfg = backbone
        self._head_cfg = head
        self._weight_decay = weight_decay

        self.get_default_attributes()
        self._loss_fn = None
        self._loss_weight = None
        self._using_rt = using_rt
        return

    def get_default_attributes(self):
        pass

    def get_summary(self):
        self._backbone.summary()
        self._head.summary()
        print(self._backbone.output_shape)
        print(self._head.output_shape)
        self.summary()
        return

    def build(self, input_shape):
        default_dict = {
            "yolov1": {
                "backbone": "yolov1_backbone",
                "head": "regular",
                "name": "yolov1"
            }
        }
        if self._backbone_cfg == None or isinstance(self._backbone_cfg, Dict):
            self._backbone_name = default_dict[self.model_name]["backbone"]
            if isinstance(self._backbone_cfg, Dict):
                default_dict[self.model_name]["backbone"] = self._backbone_cfg
            self._backbone = Backbone_Builder(
                name=default_dict[self.model_name]["backbone"],
                config=default_dict[self.model_name]["backbone"],
                input_shape=self._input_shape,
                weight_decay=self._weight_decay)
        else:
            self._backbone = self._backbone_cfg
            self._custom_aspects = True

        if self._head_cfg == None or isinstance(self._head_cfg, Dict):
            if isinstance(self._head_cfg, Dict):
                default_dict[self.model_name]["head"] = self._head_cfg
            print(self._backbone.output_shape)
            self._head = Yolov1Head(
                model=default_dict[self.model_name]["head"],
                config=default_dict[self.model_name]["head"],
                classes=self._classes,
                boxes=self._boxes,
                input_shape=self._backbone.output_shape)
        else:
            self._head = self._head_cfg
            self._custom_aspects = True

        self._model_name = default_dict[self.model_name]["name"]
        self._backbone.build(input_shape)
        print(self._backbone.output_shape, self._head.input_shape)
        self._head.build(self._backbone.output_shape)
        self._built = True
        super().build(input_shape)
        return

    def call(self, inputs, training=False):
        feature_maps = self._backbone(inputs)
        raw_head = self._head(feature_maps)
        if training or self._using_rt:
            return {"raw_output": raw_head}
        else:
            # predictions = self._head_filter(raw_head)
            predictions = raw_head # TODO: find out difference between [yolo] and [detection]
            return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone=True,
                             dn2tf_head=True,
                             config_file=None,
                             weights_file=None):
        pass

if __name__ == "__main__":
    y = Yolov1(model = "yolov1", input_shape=[1, 448, 448, 3])
    y.build(input_shape=[1, 448, 448, 3])
    y.summary()