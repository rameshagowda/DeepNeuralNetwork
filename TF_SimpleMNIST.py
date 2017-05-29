import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# use TF helper function to pul down the data from MNIST site
mnist = input_data.read_data_sets("C:\data\MNIST_data/", one_hot = True)

# x is the placeholder for 28 x 28 image data (784 pixels)
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is a 10 element vector, containing the predicted probabilities of each digit (0 - 9)
y_ = tf.placeholder(tf.float32, [None, 10])

# define weights and biases 
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define model. softmax is one of the activation function.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# use gradient descent to minimize the cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize the global variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# perform 1000 training steps
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) # do optimization with this data

# evalute how well the model did.
# actual (y) and predicted(y_)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
print("Test Accuracy : {0}%".format(test_accuracy*100))


sess.close()




