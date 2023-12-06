import tensorflow as tf

"""
Classes:
    Sequentes (no use)
    conv_block
    res_block
    Down_block
    Up_block
    Conv_7x7_64
    Conv_3
"""

"""
Functions:
    relu
    bn
    maxpool_2x2
    GroupNorm
"""
def GroupNorm(inputs, ns='GroupNorm', G = 16, eps=1e-5):
        with tf.variable_scope(ns):
            N,H,W,C = inputs.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(inputs, [-1, H, W, C//G, G])
            mean, var = tf.nn.moments(x, [1, 2, 3]) # b g
            mean = tf.reshape(mean, [-1, 1, 1, 1, G]) # b g
            var = tf.reshape(var, [-1, 1, 1, 1, G]) # b g
            x = (x-mean) / tf.sqrt(var+eps)
            # gamma,beta: scale and offset
            gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
            x = tf.reshape(x, [-1, H, W, C]) * gamma + beta
        return x

def relu(inputs):
    return tf.nn.relu(inputs)

def maxpool_2d(inputs):
    # height and weight to .5
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

class Conv_block(object):
    """Conv block for resnet"""
    def __init__(self, in_channels, namespace):
        self._ns = namespace
        with tf.name_scope(namespace):
            # conv1
            self._conv1 = tf.keras.layers.Conv2D(filters=in_channels,
                                                kernel_size = (1, 1),
                                                padding="SAME",
                                                name="conv1",
                                                strides=(1, 1))
            # conv2
            self._conv2 = tf.keras.layers.Conv2D(filters=in_channels,
                                                kernel_size = (3, 3),
                                                padding="SAME",
                                                name="conv2",
                                                strides=(1, 1))
            # conv3
            self._conv3 = tf.keras.layers.Conv2D(filters=in_channels * 2,
                                                kernel_size = (1, 1),
                                                padding="SAME",
                                                name="conv3",
                                                strides=(1, 1))
    
    def __call__(self, inputs):
        fm = self._conv1(inputs)
        fm = relu(GroupNorm(fm, ns=f"{self._ns}_gn1"))
        fm = self._conv2(fm)
        fm = relu(GroupNorm(fm, ns=f"{self._ns}_gn2"))
        fm = self._conv3(fm)
        outputs = relu(GroupNorm(fm, ns=f"{self._ns}_gn3"))
        return outputs
    

class Res_block(object):
    """The res block, linear calculation"""
    def __init__(self, in_channels, namespace):
        with tf.name_scope(namespace):
            self._conv = tf.keras.layers.Conv2D(filters=in_channels * 2,
                                                kernel_size = (3, 3),
                                                padding="SAME",
                                                name="conv",
                                                strides=(1, 1))
    
    def __call__(self, inputs):
        return self._conv(inputs)
    
    

class Up_block(object):
    
    def __init__(self, out_channels, namespace) -> None:
        self._ns = namespace
        with tf.name_scope(namespace):
            # transpos_conv
            self._convT = tf.keras.layers.Conv2DTranspose(filters=out_channels,
                                                kernel_size = (2, 2),
                                                padding="SAME",
                                                name="convT",
                                                strides=(2, 2))
            print(self._convT)
            # conv1
            self._conv1 = tf.keras.layers.Conv2D(filters=out_channels,
                                                kernel_size = (1, 1),
                                                padding="SAME",
                                                name="conv1",
                                                strides=(1, 1))
            # conv2
            self._conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                                kernel_size = (3, 3),
                                                padding="SAME",
                                                name="conv2",
                                                strides=(1, 1))
            # res block
            self._res   = tf.keras.layers.Conv2D(filters=out_channels,
                                                kernel_size = (3, 3),
                                                padding="SAME",
                                                name="res",
                                                strides=(1, 1))
    def __call__(self, ufm, dfm):
        """
        Notes:
            upsample
        Variables:
            dfm: downsample feature maps
            ufm: upsample feature maps
        """
        
        # up sample ufm
        _ufm = relu(GroupNorm(self._convT(ufm), ns=f"{self._ns}_gn1"))
        x    = tf.concat([dfm, _ufm], -1)
        fm   = relu(GroupNorm(self._conv1(x), ns=f"{self._ns}_gn2"))
        fm   = relu(GroupNorm(self._conv2(fm), ns=f"{self._ns}_gn3"))
        res = self._res(x)
        return fm + res
    
    
class Down_block(object):
    """A down blocks contains a conv_block and a res block and a pooling"""
    def __init__(self, in_channels, namspace):
        self._conv_block = Conv_block(in_channels, namespace=namspace)
        self._res_block  = Res_block(in_channels, namespace=namspace)
    
    def __call__(self, inputs):
        fm  = self._conv_block(inputs)
        res = self._res_block(inputs) 
        return fm + res
    
class Conv_7x7_64(object):
    def __init__(self, namespace):
        self._ns = namespace
        with tf.name_scope(namespace):
            self._conv = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size = (7, 7),
                                                padding="SAME",
                                                name="conv",
                                                strides=(1, 1))
            
    def __call__(self, inputs):
        return relu(GroupNorm(self._conv(inputs), ns=f"{self._ns}_gn"))

class Conv_2(object):
    """Change the feature maps to 2 chanle"""
    def __init__(self, namespace):
        with tf.name_scope(namespace):
            self._conv = tf.keras.layers.Conv2D(filters=2,
                                                kernel_size = (1, 1),
                                                padding="SAME",
                                                name="conv",
                                                strides=(1, 1))
    
    def __call__(self, inputs):
        return self._conv(inputs)