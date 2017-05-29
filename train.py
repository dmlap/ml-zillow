import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math

def input_fn(df):
    columns = {
        k: tf.constant(df[k].values)
        for k in [col.name for col in feature_columns] if not k.endswith('_bucketized')
    }
    output = tf.constant(df['logerror'].values)
    return columns, output

def fillna_df(df):
    for k in df:
        if df[k].dtype.kind in 'iuf':
            df[k].fillna(df[k].mean(), inplace=True)

print 'Reading training data...'
train_df = pd.read_csv('data/merged_train_2016.csv', parse_dates=['transactiondate'])
fillna_df(train_df)
# train_df['taxamount'].fillna(train_df['taxamount'].mean(), inplace=True)
# train_df['bathroomcnt'].fillna(train_df['bathroomcnt'].mean(), inplace=True)
# train_df['bedroomcnt'].fillna(train_df['bedroomcnt'].mean(), inplace=True)

feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount'),
    tf.contrib.layers.real_valued_column('bathroomcnt'),
    tf.contrib.layers.real_valued_column('bedroomcnt'),

    # tf.contrib.layers.sparse_column_with_hash_bucket('heatingorsystemtypeid', hash_bucket_size=100),

    tf.contrib.layers.real_valued_column('yearbuilt'),
    tf.contrib.layers.real_valued_column('calculatedfinishedsquarefeet'),
    tf.contrib.layers.real_valued_column('finishedsquarefeet12')
]
# feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[1], boundaries=range(0, 10)))
# feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[2], boundaries=range(0, 10)))
# feature_columns.append(tf.contrib.layers.crossed_column([feature_columns[3], feature_columns[4]], hash_bucket_size=100))

model = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                      model_dir='dnn-regressor-models',
                                      hidden_units=[32, 32, 16])
print 'Training...'
model.fit(input_fn=lambda: input_fn(train_df), steps=1000)

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016.csv', parse_dates=['transactiondate'])
fillna_df(eval_df)
# eval_df['taxamount'].fillna(eval_df['taxamount'].mean(), inplace=True)
# eval_df['bathroomcnt'].fillna(eval_df['bathroomcnt'].mean(), inplace=True)
# eval_df['bedroomcnt'].fillna(eval_df['bedroomcnt'].mean(), inplace=True)
results = model.evaluate(input_fn=lambda: input_fn(eval_df), steps=1000)

print 'mean squared loss: %f' % results['loss']
print 'total loss: %f' % (results['loss'] * 45407)

# print 'Sum of squared differences: %f' % overall_loss
# print 'Normalized loss: %f' % (overall_loss / len(outs))
# zipped = []
# for i in range(len(outs)):
#     zipped.append((outs[i][0], log_errors[i][0], xs[i][0], xs[i][1], xs[i][2]))

# samples = random.sample(zipped, 10)
# print 'Selected results:'
# print '{:<20}{:<20}'.format('output', 'log_error')
# for k in samples:
#     print '{:<20f}{:<20f}'.format(k[0], k[1])
