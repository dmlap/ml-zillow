import tensorflow as tf
import pandas as pd
import numpy as np

import models
tf.logging.set_verbosity(tf.logging.ERROR)
print 'Reading training data...'

train_df = pd.read_csv('data/merged_train_2016_total.csv', parse_dates=['transactiondate'])
models.fillna_df(train_df)
err_std = train_df['logerror'].std()
err_mean = train_df['logerror'].mean()
query_outl = '(logerror >= ' + str(err_std + err_mean) + ') or (logerror <= ' + str(err_mean - err_std)+ ')'
query_norm = '(logerror < ' + str(err_std + err_mean) + ') or (logerror > ' + str(err_mean - err_std) + ')'
train_df_outl = train_df.query(query_outl)
train_df_norm = train_df.query(query_norm)


#feature_columns = [
#    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float64),
#    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float64)
#]
model = models.dnn_regressor

print 'Training...'
for _ in range(1):
    print 'Iteration: %f' % (_+1)
    model.fit(input_fn=lambda: models.input_fn(train_df_outl), steps=50000)

print 'Done.'
