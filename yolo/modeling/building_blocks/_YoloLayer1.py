import tensorflow as tf
import tensorflow.keras as ks

@ks.utils.register_keras_serializable(package='yolo_v1')
class YoloLayerV1(ks.Model):
    def __init__(self, 
                 coord_scale,
                 noobj_scale,
                 num_boxes,
                 num_classes,
                 size,
                 ignore_thresh,
                 **kwargs):
        """
        Detection layer for YOLO v1
        Args: 
            coord_scale: float indicating the weight on the localization loss
            noobj_scale: float indicating the weight on the confidence loss
            num_boxes: integer, number of prediction boxes per grid cell 
            num_classes: integer, number of class probabilities each box predicts
            size: integer, specifying that the input has size * size grid cells
            ignore_thresh: float indicating the confidence threshold of whether a box
                           contains an object within it.
        call Return: 
            dict: with keys "bbox", "classes", "confidence", and "raw_output" 
        """
        super().__init__(**kwargs)
        self._coord_scale = coord_scale
        self._noobj_scale = noobj_scale
        self._num_boxes = num_boxes
        self._num_classes = num_classes
        self._ignore_thresh = ignore_thresh
        self._size = size
    
    def parse_prediction(self, inputs):
        """
        Parses a prediction tensor into its box, class, and confidence components
        Args:
            inputs: tensor of shape [batches, size, size, boxes * 5 + classes]

        Return:
            boxes, classes, and confidence tensors for the prediction 
        """

        # Seperate bounding box compoenents from class probabilities
        class_start = self._num_boxes * 5
        boxes = inputs[..., :class_start]
        classes = inputs[..., class_start:]

        # Get components from boxes:
        classes = tf.reshape(classes, [-1, 1, self._num_classes])
        boxes = tf.reshape(boxes, [-1, self._num_boxes, 5])

        boxes_xywh = boxes[..., 0:4]
        confidence = boxes[..., 4]

        # Determine class prediction:
        classes = ks.activations.softmax(classes, axis=-1)
        classes = tf.math.argmax(classes, axis=-1)

        # print(classes)
        # print(confidence)

        return None, None, None

    def call(self, inputs):
        # Reshape the input to [size, size, boxes * 5 + classes]
        desired_shape = [-1, self._size, 
                         self._size, 
                         self._num_boxes * 5 + self._num_classes]
        inputs = tf.reshape(inputs, desired_shape)
        boxes, classes, confidence = self.parse_prediction(inputs)
        print(inputs.get_shape())

        return


if __name__ == "__main__":
    num_boxes = 3
    num_classes = 20
    size = 7

    input_size = size * size * (num_boxes * 5 + num_classes)
    random_input = tf.random.uniform(shape=(1, input_size,), maxval=1, dtype=tf.float32, seed=12345)
    tf.print("Random input:", random_input)

    print("Testing yolo layer:")
    yolo_layer = YoloLayerV1(coord_scale=5.0,
                             noobj_scale=0.5,
                             num_boxes=num_boxes,
                             num_classes=num_classes,
                             size=size,
                             ignore_thresh=0.7)
    
    yolo_layer(random_input)

