import tensorflow as tf

x =tf.constant([[[[ 1, 2, 3, 4], [ 5, 6, 7, 8], [ 9, 10, 11, 12]], 
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], 
                
               [[[ 1, 2, 3, 4], [ 5, 6, 7, 8], [ 9, 10, 11, 12]], 
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])



print(x.shape)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(x).shape)

print(sess.run(tf.reduce_sum(x, axis=[1,2])))