import tensorflow as tf 
import tensorflow.keras as ks
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.utils._darknet2tf import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import load_weights_dnBackbone, split_converter
from yolo.utils.lr_schedulers import CosineDecay
from tensorflow.keras.mixed_precision import experimental as mixed_precision

'''
This script trains a simple classification head on the yolo v1 backbone
configured with pretrained weights. It uses the food101 dataset to train

'''
''' Dataset loading '''
def getDataset():
    # get classification dataset and preprocess
    DATASET_DIRECTORY = "D:\Datasets" # modify to select download location
    (train, test), info = tfds.load('mnist', 
                      split=['train', 'test'], 
                      shuffle_files=False, 
                      as_supervised=True,
                      with_info=True, 
                      data_dir=DATASET_DIRECTORY)
    train = train.map(lambda x, y: preprocess(x, y)).batch(1)
    test = test.map(lambda x, y: preprocess(x, y)).batch(1)
    return (train, test)

def preprocess(image, label):
    # some simple preprocessing: random flip and resize to 448 * 448
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.resize(image, [448, 448])

    label = tf.one_hot(label, 10, dtype = tf.float32) # for mnist
    image = tf.image.grayscale_to_rgb(image) # for mnist

    return image, label

''' Model loading '''
def loadBackboneWeights(backbone):
    config_path = "yolov1.cfg"
    weights_path = "D:/yolov1.weights"

    converter = DarkNetConverter()
    list_encdec = converter.read(config_file=config_path, weights_file=weights_path)
    encoder, _ = split_converter(list_encdec, 25)

    load_weights_dnBackbone(backbone, encoder, mtype='darknet20') # by default sets trainable to False
    backbone.trainable = False
    return

def createModel(numClasses=10):
    backbone = Backbone_Builder(name='yolov1_backbone',
                                input_shape=(None, 448, 448, 3),
                                )
    
    #loadBackboneWeights(backbone)
    backbone.summary()

    # class_model = ClassificationModel(backbone=backbone, classes=numClasses, dropout_rate=0.5)
    # x = tf.ones(shape=(1, 448, 448, 3))

    # class_model.call(x)

    class_model = ks.Sequential([
        backbone,
        ks.layers.GlobalAveragePooling2D(),
        ks.layers.Dropout(rate=0.5),
        ks.layers.Dense(numClasses),
        ks.layers.Activation(activation="softmax")
    ])

    return class_model

class ClassificationModel(ks.Model):
    def __init__(self, backbone, classes = 101, dropout_rate = 0.5, activation = "softmax", kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._classes = classes
        self._activation = activation 
        self._backbone = backbone
        self._global_avg = ks.layers.GlobalAveragePooling2D()
        self._dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self._dense_output = ks.layers.Dense(self._classes, kernel_regularizer=kernel_regularizer)
        self._activation = ks.layers.Activation(activation=self._activation)
        self._kernel_regularizer = kernel_regularizer
    
    def build(self, input_shape):
        self._backbone.build(input_shape)
        super().build(input_shape)
        return

    def call(self, inputs):
        x = self._backbone(inputs)
        l = self._global_avg(x)
        x = self._dropout(l)
        return self._activation(self._dense_output(x))
        
    def train_step(self, data):
        '''
        for float16 training
        opt = tf.keras.optimizers.SGD(0.25)
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        '''
        #get the data point
        image, label = data
        # computer detivative and apply gradients
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            # compute a prediction
            y_pred = self(image, training=True)
            loss = self.compiled_loss(label, y_pred)
            scaled_loss = loss / num_replicas
            # scale the loss for numerical stability
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)
        # compute the gradient
        train_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, train_vars)
        # get unscaled loss if the scaled_loss was used
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, train_vars))
        #custom metrics
        loss_metrics = {"loss": loss}
        self.compiled_metrics.update_state(label, y_pred)
        dict_vals = {m.name: m.result() for m in self.metrics}
        loss_metrics.update(dict_vals)
        return loss_metrics

if __name__ == '__main__':
    # Retrieve training and validation data, performs preprocessing
    (train, test) = getDataset() 
    show = False

    # Display dataset samples
    if show:
        fig, ax = plt.subplots(3, 3)
        for i, (image, label) in enumerate(test):
            ax[i // 3][i % 3].imshow(image[0,...])
            ax[i // 3][i % 3].set_xlabel(int(label))
            if i >= 8:
                break
        
        fig.tight_layout()
        plt.show()
            
    # Load the model (backbone + classification head)
    model = createModel()
    model.summary()

    learning_rate = CosineDecay(0.1, 5, 150, 0, verbose=1)

    model.compile(optimizer='sgd',
                loss=tf.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    
    epoch_history = model.fit(train, validation_data=test, epochs=1, callbacks=[learning_rate])