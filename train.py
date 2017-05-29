import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math

print 'Reading training data...'
train_df = pd.read_csv('data/merged_train_2016.csv', parse_dates=['transactiondate'])
train_df['taxamount'].fillna(train_df['taxamount'].mean(), inplace=True)
train_df['bathroomcnt'].fillna(train_df['bathroomcnt'].mean(), inplace=True)
train_df['bedroomcnt'].fillna(train_df['bedroomcnt'].mean(), inplace=True)

W = tf.Variable(tf.zeros([3, 1]), tf.float32)
b = tf.Variable(tf.zeros([1]), tf.float32)

x = tf.placeholder(tf.float32, shape=[None, 3])
output = tf.matmul(x, W) + b
log_error = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.squared_difference(output, log_error))
# optimize = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
optimize = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
print 'Training...'
for i in range(0, len(train_df), 10):
    _, _, w_, b_ = session.run([optimize, output, W, b], {
        x: zip(train_df['taxamount'].values[i:i + 10],
               train_df['bathroomcnt'].values[i:i + 10],
               train_df['bedroomcnt'].values[i:i + 10]),
        log_error: [[o] for o in train_df['logerror'].values[i:i + 10]]
    })
    if i % 1000 == 0:
        print w_, b_

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016.csv', parse_dates=['transactiondate'])
eval_df['taxamount'].fillna(eval_df['taxamount'].mean(), inplace=True)
eval_df['bathroomcnt'].fillna(eval_df['bathroomcnt'].mean(), inplace=True)
eval_df['bedroomcnt'].fillna(eval_df['bedroomcnt'].mean(), inplace=True)
outs, log_errors, xs, overall_loss = session.run([output, log_error, x, loss], {
    x: zip(eval_df['taxamount'].values,
           eval_df['bathroomcnt'].values,
           eval_df['bedroomcnt'].values),
    log_error: [[o] for o in eval_df['logerror'].values]
})


print 'Sum of squared differences: %f' % overall_loss
print 'Normalized loss: %f' % (overall_loss / len(outs))
zipped = []
for i in range(len(outs)):
    zipped.append((outs[i][0], log_errors[i][0], xs[i][0], xs[i][1], xs[i][2]))

samples = random.sample(zipped, 10)
print 'Selected results:'
print '{:<20}{:<20}'.format('output', 'log_error')
for k in samples:
    print '{:<20f}{:<20f}'.format(k[0], k[1])


