import tensorflow as tf
import sys
sys.path.append("..")
from models.layers import *
import numpy as np
from models.functions import *


"""
Classes:
    U_net
"""

class U_net(object):
    
    
    def __init__(self):
        """
        Inputs are normalization
        """
        # down_side
        self._c7    = Conv_7x7_64("p") # no pooling
        self._down1 = Down_block(64, "d1") #  256,  256, 64 (inputs)
        self._down2 = Down_block(128, "d2") #  128,  128, 128
        self._down3 = Down_block(256, "d3") #  64,   64,  256
        self._down4 = Down_block(512, "d4") # 32,   32,  512
        self._down5 = Down_block(1024, "d5") # 16,   16,  1024
        self._downe = Down_block(2048, "de") # 8,    8,   2048
        
        # up_side
        self._up1 = Up_block(1024, "u1") # 16,  16,  1024 (outputs)
        self._up2 = Up_block(512, "u2") # 32,  32,  512
        self._up3 = Up_block(256, "u3") #  64,  64,  256
        self._up4 = Up_block(128, "u4") #  128, 128, 128
        self._up5 = Up_block(64, "u5") #  256, 256, 64
        
        self._conv_3 = Conv_2("c2")
        self._flatten = tf.layers.Flatten()
    def __call__(self, inputs):
        # Down side
        dfm1 = self._down1(inputs)
        dfm2 = self._down2(maxpool_2d(dfm1))
        dfm3 = self._down3(maxpool_2d(dfm2))
        dfm4 = self._down4(maxpool_2d(dfm3))
        dfm5 = self._down5(maxpool_2d(dfm4))
        dfme = self._downe(maxpool_2d(dfm5))
        print(dfme.shape)
        # Up side
        ufm1 = self._up1(dfm=dfm5, ufm=dfme)
        ufm2 = self._up2(dfm=dfm4, ufm=ufm1)
        ufm3 = self._up3(dfm=dfm3, ufm=ufm2)
        ufm4 = self._up4(dfm=dfm2, ufm=ufm3)
        ufm5 = self._up5(dfm=dfm1, ufm=ufm4)
        fm = tf.nn.softmax(self._conv_3(ufm5), axis=-1)
        return fm

if __name__ == "__main__":
    net = U_net()
    x_p = tf.placeholder(tf.float32, [None, 256, 256, 2])
    xs = np.random.random([12, 256, 256, 2])
    o = net(x_p)
    with tf.Session() as sess:
        init_var(sess)
        _o = sess.run(o, feed_dict={x_p:xs})
        print(_o.shape)
        print(_o[0,0,0,:])
        exit()
