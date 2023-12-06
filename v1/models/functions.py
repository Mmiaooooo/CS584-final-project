import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Functions:
    init_var
    ce_loss_v1
    dice_loss_v1
    accuracy
"""


def init_var(sess):
    inits = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(inits)

def ce_loss_v1(probs, mask, n_class=2, weights=None):
    # change to 1d
    with tf.name_scope('bce_loss'):
        mask = tf.squeeze(mask)
        
        # weights
        pass
        masks = tf.cast(mask, tf.int32)
        masks = tf.to_float(tf.one_hot(masks, n_class))
        
        # print('masks   = ', masks)
        # print('probs   = ', probs)
        #probs = tf.clip_by_value(probs,1e-10,1.0)
        loss = -tf.reduce_sum(masks * tf.log(probs + 1e-10), axis=1)
        return tf.reduce_mean(loss)

def dice_loss_V1(probs, masks, n_class=2):
    with tf.name_scope('dice_loss'):
        eps = 1e-6
        dice = []
        # weights = [1 / n_class for x in range(n_class)]
        masks = tf.one_hot(masks, n_class)
        masks = tf.cast(masks, tf.float32)
        for c in range(1, n_class):
            intersection = 2 * tf.reduce_sum(tf.multiply(probs[..., c], masks[..., c]), axis=[1, 2])
            surfacesum = tf.add(tf.reduce_sum(masks[..., c], axis=[1, 2]), tf.reduce_sum(probs[..., c], axis=[1, 2]))
            dice.append(tf.divide(intersection, (surfacesum + eps)))
            # print('dice = ', dice)
        # dice_weighted = tf.reduce_sum(tf.multiply(weights,dice),axis=-1)

    return 1 - tf.reduce_mean(dice)

def accuracy(preds, gts):
    """
    Assume preds and gts are flatten
    """
    correct_preds = tf.equal(preds, gts)
    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))

def save_example(imgs, name):
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    ax[0].imshow(imgs[0])
    ax[1].imshow(imgs[1])
    ax[2].imshow(imgs[2])
    plt.title(name)
    plt.savefig(rf'pics{os.sep}{name}.png')
    plt.close()