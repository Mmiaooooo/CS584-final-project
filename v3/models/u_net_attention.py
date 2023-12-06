import tensorflow as tf
import numpy as np

class UNet(object):
    
    def __init__(self, seg_class = 3, depth=4, filt=32):
        self._seg_class     = seg_class
        self._depth         = depth
        self._filter        = filt
        # some classes
        pass
        
    def MaxPooling2d(self, inputs, ps=2, s=2, p='same'):
        # pool_size is kernel size
        return self._MaxPooling(inputs, pool_size=ps, strides=s, padding=p)
    
    def AvgPooling2d(self, inputs, ps=2, s=2, p='same'):
        return self._AvgPooling(inputs, pool_size=ps, strides=s, padding=p)
    
    def GroupNorm(self, inputs, ns='GroupNorm', G = 32, eps=1e-5):
        with tf.variable_scope(ns):
            pass
    
    def Conv2d(self, inputs, filters, name_scope='Conv2d', ks=3, s=1, p='same'):
        # no res block
        with tf.variable_scope(name_scope):
            pass
    
    def Conv2dTranspose(self, inputs, filters, ks=3, s=2, p='same'):
        return self._ConvTranspose(inputs, filters=filters, kernel_size=ks, strides=s, padding=p)
    
    def Conv2d_Block(self, x, filters, ns='Conv2d_Block', n_layers=1, ks=3, s=1, p='same'):
        with tf.variable_scope(ns):
            pass
    
    def Res2d_Block(self, inputs, filters, ns='Res2d_Block', n_layers=1, ks=3, s=1, p='same'):
        # change the inputs
        with tf.variable_scope(ns):
            pass
        
    def Basic_Block(self, inputs, filters, conv=None, ns='', n_layer=1, ks=3, s=1, p='same'):
        if conv == 'Res' or conv == 'res':
            return self.Res2d_Block(inputs=inputs, filters=filters, ns='Res2d_Block'+ns, n_layers=n_layer, ks=ks, s=s, p=p)
        else:
            return self.Conv2d_Block(inputs, filters, 'Conv2d_Block'+ns, n_layer, ks, s, p)
    
    '''
    Unet --- encoder
    '''
    
    def Unet_encoder_v1(self, inputs, conv='Res'):
        
        features = list()
        filters = [32, 64, 128, 256, 512]
        x = tf.identity(inputs, name='inputs')
        
        with tf.variable_scope('UNet_encoder', reuse=tf.AUTO_REUSE):
            pass
    
    def Unet_decoder(self, encoder_out, encoder_featrues, conv=None):
        pass
    
    def Unet_mainBody(self, inputs, conv):
        encoder_out, encoder_out_features = self.Unet_encoder_v1(inputs, conv=conv)
        decoder_out_seg = self.Unet_decoder(encoder_out, encoder_out_features, conv=conv)
        softmax_seg = tf.nn.softmax(decoder_out_seg, axis=-1)
        return softmax_seg

    def lnLayer(self,inputs, axis=-1, ns='Ln', eps=1e-8):
        pass
    def batchDrop(self,tensor, batch_dropRate=0., training=False, ns='BatchDrop'):
        pass
    
    '''
    Transformer and attention
    '''
    def transfomer_attention_encode(self,cls_vector, word_count, word_length, num_heads):
        # cls_vector_query = tf.reshape(cls_vector, (-1, word_count, vector_channel // word_count))
        pass
    def multiHeadAttention(self,
                           encoder_vec,
                           numHead       = 8,  # 多头注意力数量
                           ns            ='attention',
                           dropRate1     = 0.5,
                           dropBatchRate = 0.5,
                           training      = False):
        """
        @param queries: input shape (B * N * C) or (B * C)
        @param numHead: num of multi heads
        @param ns: name scope
        @return: (B * N * C) or (B * C)
        """
        pass

    def tranformer_ff_layer(self,vector_input, num_units, scope):
        pass
    def ffLayer(self,queries, num_units, ns='Ff'):
        pass
    #     trans_numBlock = 2
    #     trans_numHeads = 8
    #     trans_numFfunit = 256
    #     trans_dropRate = 0.5
    #     trans_dropBatchRate = 0.5
    def Transformer_encode(self,encoder_vec,
                          trans_numBlock,
                          trans_numHeads,
                          trans_numFfunit,
                          trans_dropRate,
                          trans_dropBatchRate,
                          ns='SimpleTransformer',
                          training=False):
        pass
    def fmap_2_vec_V1(self, inputs_1, inputs_2, inputs_3, inputs_4, inputs_5):
        pass
    def fmap_2_vec(self, inputs, num):
        pass


if __name__ == "__main__":
    # test ...
    B = 9
    H = 128
    W = 128
    C = 3
    #
    a = np.ones((B,H,W,C))
    inputs_tensor = tf.convert_to_tensor(a, dtype=tf.float32)
    model = UNet(seg_class=2, filt=32)
    seg_probs = model.Unet_mainBody(inputs_tensor, conv='res')
    seg_preds_img = tf.cast(tf.argmax(seg_probs, -1), tf.int32)
    print('seg_probs = ', seg_probs)
    print('seg_imgs= ', seg_preds_img)