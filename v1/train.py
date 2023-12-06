from models.functions import *
from models.u_net import *
import tensorflow as tf
from datas.read_data import *
import sys

def save(sess, saver, epoch):
    print(f'Saving:{epoch + 1}')
    saver.save(sess, f"my_nets/UNET_{epoch + 1}.ckpt")

# 获取所有可用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

if __name__ == "__main__":
    lr = 0.00001
   
    if int(sys.argv[1]) == 0:

        batch_size = 64
        num_epochs = 10
        TRAIN_PATH = r"D:\dl\datas\skin18\train_data"
        TRAIN_GT_PATH = r"D:\dl\datas\skin18\train_gt"
        TEST_PATH = r"D:\dl\datas\skin18\test_data"
        TEST_GT_PATH = r"D:\dl\datas\skin18\test_gt"
        
        
    elif int(sys.argv[1]) == 1:
        TRAIN_PATH = r"/home/miaomukang/datasets/isic18/train_data"
        TRAIN_GT_PATH = r"/home/miaomukang/datasets/isic18/train_gt"
        TEST_PATH = r"/home/miaomukang/datasets/isic18/test_data"
        TEST_GT_PATH = r"/home/miaomukang/datasets/isic18/test_gt"
        batch_size = 64
        num_epochs = 150

    train_names = read_file_names(TRAIN_PATH)
    train_gt_names = read_file_names(TRAIN_GT_PATH)
    print(f'Read data finish, num train:{len(train_names)}, num test: {len(train_gt_names)}')        
    train_datas = build_dataset(train_names, train_gt_names)
    tg = DataGenerator(train_datas)  

    test_names = read_file_names(TEST_PATH)
    test_gt_names = read_file_names(TEST_GT_PATH)
    print(f'Read data finish, num train:{len(test_names)}, num test: {len(test_gt_names)}')        
    test_datas = build_dataset(test_names, test_gt_names)
    ttg = DataGenerator(test_datas)  
    
    # find
    f85 = False
    f90 = False
    
    net     = U_net()
    flatten = tf.layers.Flatten()
    # placeholder
    x_p = tf.placeholder(tf.float32, [None, 128, 128, 3])
    y_p = tf.placeholder(tf.int32,   [None, 128, 128])
    t_p = tf.placeholder(tf.bool)
    probs = net(x_p)
    # nodes
    preds = tf.cast(tf.argmax(probs, -1), tf.int32)
    ce_loss = ce_loss_v1(tf.reshape(probs, (-1, 128*128, 2)), flatten(y_p), n_class=2)
    dice_loss = dice_loss_V1(probs, y_p, 2)

    # loss 
    loss = ce_loss + dice_loss
    # acc
    acc = accuracy(flatten(preds), flatten(y_p))
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(lr,global_step, 250, 0.97, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=20)
    # train
    with tf.Session(config=config) as sess:
        init_var(sess)
        for epoch in range(num_epochs):
            trainIter = get_train_generator(tg, batch_size=batch_size)
            for (xs, ys), bi, names in trainIter:
                if (bi + 1) % 10 == 0 or bi == 0:
                    print(f"Unique y Values for {bi}:", np.unique(ys))
                _, _loss, _acc, _preds, _dice = sess.run([train_step, loss, acc, preds, dice_loss], feed_dict={x_p:xs, y_p:ys})
                if (bi + 1) % 3 ==0 or bi == 0:
                    print(f"Epoch: {epoch}, loss: {_loss}, acc: {_acc}, dice{1 - _dice}")
                if (bi + 1) % 10 ==0 or bi == 0:
                    name0 = parse_gt_name(names[0]['y'])
                    save_example([xs[0], _preds[0], ys[0]], name0)

            # test
            test_correct = 0
            test_loss    = 0
            test_nums    = 0
            test_dice    = 0
            validIter = get_train_generator(ttg, batch_size=8)
            for (xs, ys), bi, names in validIter:
                _loss, _acc, _preds, _dice = sess.run([loss, acc, preds, dice_loss], feed_dict={x_p:xs, y_p:ys})
                test_loss += _loss
                test_correct += _acc
                test_dice += _dice
                test_nums += 1
            
            test_dice = 1 - test_dice / test_nums
            print(f'Test Dice: {test_dice} \n', '=' * 80)
            if test_dice > 0.85 and not f85:
                save(sess, saver, epoch)
                f85 = True
                continue
            
            if test_dice > 0.90 and not f90:
                save(sess, saver, epoch)
                f90 = True
                continue
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                save(sess, saver, epoch)