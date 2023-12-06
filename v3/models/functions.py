import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


sep = os.sep
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
    print(probs, mask, "+"*60)
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
        pass

  

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
    print('save', f'pics{os.sep}{name}.png')
    plt.savefig(f'pics{os.sep}{name}.png')
    plt.show()
    exit()
    plt.close()

def save_train_example(img, gt, masks, name):
    fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    ax[0][0].imshow(img)
    ax[0][0].set_title('Original image')
    ax[0][1].imshow(gt)
    ax[0][1].set_title('GT')
    ax[0][2].imshow(masks)
    ax[0][2].set_title('Mask')
    img_c = img.copy()
    for c in range(img_c.shape[-1]):
        img_c[...,c][masks == 1] = 125
        img_c[...,c][gt == 1] = 255
    ax[1][0].imshow(img_c)
    ax[1][0].set_title('gt=255')
    
    img_c = img.copy()
    for c in range(img_c.shape[-1]):
        img_c[...,c][gt == 1] = 125
        img_c[...,c][masks == 1] = 255
    ax[1][1].imshow(img_c)
    ax[1][1].set_title('mask=255')
    plt.savefig(rf'tpics{os.sep}{name}.png')
    plt.close()
    
def save_test_example(img, gt, masks, name):
    fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    ax[0][0].imshow(img)
    ax[0][0].set_title('Original image')
    ax[0][1].imshow(gt)
    ax[0][1].set_title('GT')
    ax[0][2].imshow(masks)
    ax[0][2].set_title('Mask')
    img_c = img.copy()
    for c in range(img_c.shape[-1]):
        img_c[...,c][masks == 1] = 125
        img_c[...,c][gt == 1] = 255
    ax[1][0].imshow(img_c)
    ax[1][0].set_title('gt=255')
    
    img_c = img.copy()
    for c in range(img_c.shape[-1]):
        img_c[...,c][gt == 1] = 125
        img_c[...,c][masks == 1] = 255
    ax[1][1].imshow(img_c)
    ax[1][1].set_title('mask=255')
    plt.savefig(rf'pics{os.sep}{name}.png')
    plt.close()
    

def save_dices(dices, mode='train'):
    assert mode == 'train' or mode == 'test', mode
    lend = len(dices)
    idxs = list(range(lend))
    save_path = f'pics{sep}{mode}{sep}dice{sep}{lend}.png'
    plt.plot(idxs, dices)
    plt.ylabel('Dice')
    plt.title(f'{mode}_{lend}_Dice')
    plt.savefig(save_path)
    plt.close()
    if len(dices) > 10:
        save_path = f'pics{sep}{mode}{sep}dice{sep}{lend}cut.png'
        plt.plot(idxs[10:], dices[10:])
        plt.ylabel('Dice')
        plt.title(f'{mode}_{lend}_Dice')
        plt.savefig(save_path)
        plt.close()
    
def save_ces(ces, mode='train'):
    assert mode == 'train' or mode == 'test', mode
    lend = len(ces)
    idxs = list(range(lend))
    save_path = f'pics{sep}{mode}{sep}ce{sep}{lend}.png'
    plt.plot(idxs, ces)
    plt.ylabel('CE')
    plt.title(f'{mode}_{lend}_CE')
    plt.savefig(save_path)
    plt.close()

def save_losses(losses, mode='train'):
    assert mode == 'train' or mode == 'test', mode
    lend = len(losses)
    idxs = list(range(lend))
    save_path = f'pics{sep}{mode}{sep}loss{sep}{lend}.png'
    plt.plot(idxs, losses)
    plt.ylabel('Loss')
    plt.title(f'{mode}_{lend}_Loss')
    plt.savefig(save_path)
    plt.close()
    