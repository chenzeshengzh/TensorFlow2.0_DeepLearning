import tensorflow as tf
from tensorflow.keras import Sequential, layers


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, kernel_size=3, strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filter_num, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, kernel_size=1, strides=strides, padding='same'))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = self.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        ])
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], 2)
        self.layer3 = self.build_resblock(256, layer_dims[2], 2)
        self.layer4 = self.build_resblock(512, layer_dims[3], 2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, strides=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, strides))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, strides=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
