import tensorflow as tf 
import tensorflow.keras as ks
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.utils._darknet2tf import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import load_weights_dnBackbone, split_converter

''' Dataset loading '''
def getDataset():
    # get img classification dataset and preprocess
    DATASET_DIRECTORY = "D:\Datasets" # modify to select download location
    (train, test), info = tfds.load('food101', 
                      split=['train', 'validation'], 
                      shuffle_files=False, 
                      as_supervised=True,
                      with_info=True, 
                      data_dir=DATASET_DIRECTORY)
    train = train.map(lambda x, y: preprocess(x, y)).batch(40)
    test = test.map(lambda x, y: preprocess(x, y)).batch(40)
    return (train, test)

def preprocess(image, label):
    # some simple preprocessing: 
    # random flip
    # resize to 448 * 448
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [448, 448])

    return image, label

''' Model loading '''
def createModel(numClasses=101):
    backbone = Backbone_Builder(name='yolov1_backbone',
                                input_shape=(None, 448, 448, 3),
                                )
    
    loadBackboneWeights(backbone)

    model = ks.Sequential([
        backbone,
        ks.layers.Flatten(),
        ks.layers.Dropout(rate=0.5),
        ks.layers.Dense(numClasses)
    ])

    tf.print(backbone.summary())
    return model

def loadBackboneWeights(backbone):
    config_path = "yolo/utils/_darknet2tf/test_locally_connected_config.cfg"
    weights_path = "D:/yolov1.weights"

    converter = DarkNetConverter()
    list_encdec = converter.read(config_file=config_path, weights_file=weights_path)
    encoder, _ = split_converter(list_encdec, 25)


    load_weights_dnBackbone(backbone, encoder, mtype='darknet20')
    return

def getWeights(backbone, encoder):
    # two darkconvs per route process
    for layer in encoder:
        tf.print(type(layer))
    return 0

if __name__ == '__main__':
    (train, test) = getDataset()
    show = False

    if show:
        fig, ax = plt.subplots(3, 3)
        for i, (image, label) in enumerate(test):
            ax[i // 3][i % 3].imshow(image[0,...])
            ax[i // 3][i % 3].set_xlabel(int(label))
            if i >= 8:
                break
        
        fig.tight_layout()
        plt.show()
            

    model = createModel()
    tf.print(model.summary())

    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    model.fit(train, validation_data=test, epochs=200)

