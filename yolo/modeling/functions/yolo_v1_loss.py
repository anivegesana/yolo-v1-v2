import tensorflow as tf
import tensorflow.keras as ks

from yolo.utils.iou_utils import *

class Yolo_Loss_v1(ks.losses.Loss):
    def __init__(self,
                 coord_scale=5.0,
                 noobj_scale=0.5,
                 num_boxes=2,
                 num_classes=20,
                 ignore_thresh=0.7,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name=None,
                 **kwargs):
        """
        Detection loss function parameters (YOLO v1)

        Args: 
            coord_scale: float indicating the weight on the localization loss
            noobj_scale: float indicating the weight on the confidence loss
            num_boxes: integer, number of prediction boxes per grid cell 
            num_classes: integer, number of class probabilities each box predicts
            ignore_thresh: float indicating the confidence threshold of whether a box
                           contains an object within it.
        call Return: 
            float: for the average loss 
        """
        super(Yolo_Loss_v1, self).__init__(reduction=reduction,
                                           name=name,
                                           **kwargs)
        self._coord_scale = coord_scale
        self._noobj_scale = noobj_scale
        self._num_boxes = num_boxes
        self._num_classes = num_classes
        self._ignore_thresh = ignore_thresh

        # metrics
        self._localization_loss = 0.0
        self._confidence_loss = 0.0
        self._classification_loss = 0.0
        
    
    def call(self, y_true, y_pred):        
        class_start = self._num_boxes * 5

        # Seperate bounding box components from class probabilities
        pred_boxes = y_pred[..., :class_start]
        pred_class = y_pred[..., class_start:]

        true_boxes = y_true[..., :class_start]
        true_class = y_true[..., class_start:]

        # Get components from the box
        pred_boxes = tf.reshape(pred_boxes, [-1, self._num_boxes, 5])
        true_boxes = tf.reshape(true_boxes, [-1, self._num_boxes, 5])

        pred_boxes_xywh = pred_boxes[..., 0:4] 
        pred_xy = pred_boxes_xywh[..., 0:2]
        pred_wh = pred_boxes_xywh[..., 2:4]
        pred_confidence = pred_boxes[..., 4]

        true_boxes_xywh = true_boxes[..., 0:4] 
        true_xy = true_boxes_xywh[..., 0:2]
        true_wh = true_boxes_xywh[..., 2:4]
        true_confidence = true_boxes[..., 4]

        # Determine IOU of all predictor boxes vs gt boxes in each cell
        iou = compute_iou(true_boxes_xywh, pred_boxes_xywh)

        # Mask off the non-predictor bounding boxes based on iou
        predictor_mask = self.get_predictor_mask(iou)

        # Localization loss:
        loss_xy = tf.reduce_sum(tf.math.square(true_xy - pred_xy), axis=-1) * predictor_mask * true_confidence
        loss_wh = tf.reduce_sum(tf.math.square(tf.math.sqrt(true_wh) -
                                         tf.math.sqrt(pred_wh)), axis=-1) * predictor_mask * true_confidence

        localization_loss = self._coord_scale * (tf.reduce_mean(loss_xy + loss_wh))

        # Confidence loss:
        obj_loss = tf.math.square(true_confidence - 
                            pred_confidence) * predictor_mask * true_confidence
        noobj_loss = tf.math.square(true_confidence - 
                              pred_confidence) * (1 - predictor_mask) * (1 - true_confidence)
        
        confidence_loss = tf.reduce_mean(obj_loss + self._noobj_scale * noobj_loss)

        # Class Probability loss:
        # TODO: Need to implement
        classification_loss = 0

        # Update metrics:
        self._localization_loss = localization_loss
        self._confidence_loss = confidence_loss
        self._classification_loss = classification_loss

        # Final loss:
        return localization_loss + confidence_loss + classification_loss
    
    def get_predictor_mask(self, iou):
        """
        Generates a mask for each cell indicating which box among the predictions should
        be used when computing localization and confidence loss.

        The predictor mask is generated by finding the argmax of the prediction box to ground
        truth IOU. In the case of ties, the smallest index is chosen due to nature of tf.math.argmax

        Args: 
            iou: Tensor of shape [num_cells, num_boxes] denoting the iou of each box compared to the
                 ground truth of each cell
        Return: 
            Tensor of shape [num_cells, num_boxes] denoting the boxes to be used as predictors
        """
        shape = iou.get_shape().as_list()
        num_cells = shape[0]
        num_boxes = shape[1]

        predictor_mask = []
        for i in range(num_cells):
            max_iou_idx = tf.math.argmax(iou[i])
            mask_val = [0] * num_boxes
            mask_val[max_iou_idx] = 1
            predictor_mask.append(tf.convert_to_tensor(mask_val, dtype=tf.float32))

        return tf.stack(predictor_mask)

    # NOTE: OLD PREDICTOR MASK GENERATOR
    # def get_predictor_mask(self, iou, penalize_miss=False):
    #     # If penalize_miss is True, then if both bounding box iou's are 0
    #     # (ie. both bounding boxes completely miss the ground truth box),
    #     # then both boxes are considered in the loss
    
    #     highest_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
    #     if penalize_miss:
    #         # The box with the highest iou is assigned 1, the other 0
    #         # If both boxes have iou 0 or tie, they are both assigned 1 
    #         highest_iou_mask = iou >= highest_iou
    #     else:
    #         # The box with the highest iou is assigned 1, the other 0
    #         # If both boxes have iou 0, they are both assigned 0. If they
    #         # are not 0 and tie, then they are both assigned 1
    #         highest_iou_mask = tf.math.divide_no_nan(iou, highest_iou)
    #         highest_iou_mask = highest_iou_mask >= 1
        
    #     return tf.cast(highest_iou_mask, dtype=tf.float32)