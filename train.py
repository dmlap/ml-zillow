import tensorflow as tf
import pandas as pd
import numpy as np
import random

print 'Reading training data...'
train_df = pd.read_csv('data/merged_train_2016.csv', parse_dates=['transactiondate'])
train_df['bathroomcnt'].fillna(train_df['bathroomcnt'].mean(), inplace=True)
train_df['bedroomcnt'].fillna(train_df['bedroomcnt'].mean(), inplace=True)

W = tf.Variable(tf.zeros([2, 1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)

x = tf.placeholder(tf.float32, shape=[None, 2])
output = tf.matmul(x, W) + b
log_error = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.squared_difference(output, log_error))
optimize = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# [?, 2] * [2, 1] => [?, 1]
# [?, 1] + [?, 1] => [?, 1]

session = tf.Session()
session.run(tf.global_variables_initializer())
print 'Training...'
session.run([optimize, W, b], {
    x: zip(train_df['bathroomcnt'].values[0:20], 
           train_df['bedroomcnt'].values[0:20]),
    log_error: [[o] for o in train_df['logerror'].values[0:20]]
})

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016.csv', parse_dates=['transactiondate'])
outs, log_errors, overall_loss = session.run([output, log_error, loss], {
    x: zip(eval_df['bathroomcnt'].values[0:20],
           eval_df['bedroomcnt'].values[0:20]),
    log_error: [[o] for o in eval_df['logerror'].values[0:20]]
})

print 'Sum of squared differences: %f' % overall_loss
samples = random.sample(zip(np.ndarray.flatten(outs), np.ndarray.flatten(log_errors)), 10)
print 'Selected results:'
print '{:<10}{:<10}'.format('output', 'log_error')
for k in samples:
    print '{:<10f}{:<10f}'.format(k[0], k[1])


