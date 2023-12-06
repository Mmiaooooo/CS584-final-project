from models.functions import *
from models.u_net import *
import tensorflow as tf
from datas.get_data import *
import sys
        
        
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def save(sess, saver, epoch):
    print(f'Saving:{epoch + 1}')
    saver.save(sess, f"my_nets/UNET_{epoch + 1}.ckpt")



if __name__ == "__main__":
    # lr = 0.0001
    lr = 0.0001
   
    mkdir('pics')
    mkdir(f'pics{sep}train')
    mkdir(f'pics{sep}train{sep}dice')
    mkdir(f'pics{sep}train{sep}loss')
    mkdir(f'pics{sep}train{sep}ce')

    mkdir(f'pics{sep}test')
    mkdir(f'pics{sep}test{sep}dice')
    mkdir(f'pics{sep}test{sep}loss')
    mkdir(f'pics{sep}test{sep}ce')
    
    if int(sys.argv[1]) == 0:
        batch_size = 4
        num_epochs = 10
        TRAIN_SAVE_PATH = r"D:\dl\datas\skin18\train"
        TEST_SAVE_PATH = r"D:\dl\datas\skin18\test"
        train_data = DataGenerator(TRAIN_SAVE_PATH, nums=8)
        test_data = DataGenerator(TEST_SAVE_PATH, is_train=False, nums=8)
            
    elif int(sys.argv[1]) == 1:
        # 获取所有可用GPU
        TRAIN_SAVE_PATH = r"/home/miaomukang/datasets/isic18/train"
        TEST_SAVE_PATH = r"/home/miaomukang/datasets/isic18/test"
        batch_size = 48
        num_epochs = 400
        train_data = DataGenerator(TRAIN_SAVE_PATH)
        test_data = DataGenerator(TEST_SAVE_PATH, is_train=False)


    
    net_saver = NetSaver([0.85, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895, 0.9, 0.91], save_nums=3)
    
    # find
    f85 = False
    f90 = False
    
    net     = UNet(seg_class=2, filt=32)
    flatten = tf.layers.Flatten()
    # placeholder
    x_p = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y_p = tf.placeholder(tf.int32,   [None, 256, 256])
    t_p = tf.placeholder(tf.bool)
    probs = net.Unet_mainBody(x_p, conv='res')
    # nodes
    preds = tf.cast(tf.argmax(probs, -1), tf.int32)
    ce_loss = ce_loss_v1(tf.reshape(probs, (-1, 256*256, 2)), flatten(y_p), n_class=2)
    dice_loss = dice_loss_V1(probs, y_p, 2)

    # loss 
    loss = ce_loss + dice_loss
    # acc
    acc = accuracy(flatten(preds), flatten(y_p))
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(lr,global_step, 250, 0.97, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=40)
    
    # records 
    train_ces = []
    train_dices = []
    train_losses = []
    
    test_ces = []
    test_dices = []
    test_losses= []
    
    # train
    with tf.Session(config=config) as sess:
        init_var(sess)
        for epoch in range(num_epochs):            
            train_loss    = 0.
            train_dice    = 0.
            train_ce      = 0.
            train_nums    = 0
            
            trainIter = train_data.get_DataIter(batch_size=batch_size)
            for xs, ys, bi, names in trainIter:
                if (bi + 1) % 10 == 0 or bi == 0:
                    print(f"Unique y Values for {bi}:", np.unique(ys))
                _, _loss, _acc, _preds, _dice, _ce = sess.run([train_step, loss, acc, preds, dice_loss, ce_loss], feed_dict={x_p:xs, y_p:ys})
                train_loss  += _loss
                train_ce    += _ce
                train_dice  += _dice
                train_nums  += 1
                if (bi + 1) % 3 ==0 or bi == 0:
                    print(f"Epoch: {epoch}, loss: {_loss}, acc: {_acc}, dice: {1 - _dice}, ce: {_ce}")
                    
                if (bi + 1) % 10 ==0 or bi == 0:
                    name0 = names[0]
                    save_train_example(xs[0], ys[0], _preds[0], name0)

            train_dice = 1 - train_dice / train_nums
            train_ce   = train_ce  / train_nums
            train_loss = train_loss / train_nums

            train_ces.append(train_ce)
            train_dices.append(train_dice)
            train_losses.append(train_loss)
            
            save_losses(train_losses)        
            save_ces(train_ces)    
            save_dices(train_dices)        
            
            print(f'Train Loss: {train_loss} \n')
            print(f'Train Dice: {train_dice} \n')
            
            # test
            test_correct = 0.
            test_loss    = 0.
            test_nums    = 0
            test_dice    = 0.
            test_ce      = 0.
            testIter = test_data.get_DataIter(batch_size=batch_size)
            for xs, ys, bi, names in testIter:
                _loss, _acc, _preds, _dice, _ce = sess.run([loss, acc, preds, dice_loss, ce_loss], feed_dict={x_p:xs, y_p:ys})
                test_loss += _loss
                test_correct += _acc
                test_dice += _dice
                test_nums += 1
                test_ce += _ce
            
            test_dice = 1 - test_dice / test_nums
            test_ce = test_ce / test_nums
            test_loss = test_loss / test_nums
            
            test_ces.append(test_ce)
            test_dices.append(test_dice)
            test_losses.append(test_loss)
            
            save_losses(test_losses, mode='test')        
            save_ces(test_ces, mode='test')    
            save_dices(test_dices, mode='test')     
            
            print(f'Test Loss: {test_loss}')
            print(f'Test Dice: {test_dice} \n', '=' * 80)
            
            is_save_net = net_saver.check_value(test_dice)
            if is_save_net:
                save(sess, saver, epoch)
                continue
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                save(sess, saver, epoch)