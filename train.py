import tensorflow as tf
import pandas as pd
import numpy as np

import models

print 'Reading training data...'

train_df = pd.read_csv('data/merged_train_2016.csv', parse_dates=['transactiondate'], nrows=50)
models.fillna_df(train_df)
feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32),
    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32)
]

model = models.dnn_regressor

print 'Training...'
for _ in range(10):
    model.fit(input_fn=lambda: models.input_fn(train_df), steps=5)

print 'Done.'
