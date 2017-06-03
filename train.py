import tensorflow as tf
import pandas as pd
import numpy as np
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
        if df[k].dtype.kind in 'iufc':
            df[k].fillna(df[k].mean() if not math.isnan(df[k].mean()) else 0, inplace=True)

print 'Reading training data...'

train_df = pd.read_csv('data/merged_train_2016.csv', parse_dates=['transactiondate'], nrows=50)
fillna_df(train_df)
feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32),
    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32)
]

model = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                      model_dir = './dnn-regressor-model',
                                      hidden_units = [4],
                                      dropout = 0,
                                      activation_fn = tf.nn.relu,
                                      optimizer=tf.train.AdagradOptimizer(learning_rate=.3),
                                      enable_centered_bias=True,
                                      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

print 'Training...'
for _ in range(10):
    model.fit(input_fn=lambda: input_fn(train_df), steps=5)

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016.csv', parse_dates=['transactiondate'], nrows=50)
fillna_df(eval_df)
results = model.evaluate(input_fn=lambda: input_fn(train_df), steps=5)

print results

input_samples = train_df.sample(n=10)
output_samples = list(model.predict(input_fn=lambda: input_fn(input_samples),
                                    outputs=None))
print 'Selected Results'
print '{:<20}{:<20}'.format('prediction', 'actual')
for k in range(len(output_samples)):
    print '{:<20f}{:<20f}'.format(output_samples[k], input_samples['logerror'].values[k])
