import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math
import time
start_time = time.time()
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.INFO)
def input_fn(df):
    print 'input_fn'
    #columns = {
    #    k: tf.constant(df[k].values)
    #    for k in [col.name for col in feature_columns] if not (k.endswith('_bucketized') or #k.endswith('_one_hot'))
    #}
    columns = {}
    #print [col.name for col in feature_columns]
    for k in [col.name for col in feature_columns]:
        #if not (k.endswith('_bucketized') or k.endswith('_one_hot')):

            #print k
            #print df[k].values
            #print df.values
            columns[k] = tf.constant(df[k].values)


    #print 'columns'
    output = tf.constant(df['logerror'].values)
    print columns
    print 'output:'
    print output
    return columns, output

def fillna_df(df):
    for k in df:
        #print df[k].dtype.kind[0]
        #print df[k].dtype.kind
        if df[k].dtype.kind[0] in 'fc':
            #print 'casting'
            #print df[k].name
            df[k]=df[k].astype('float32')

        if df[k].dtype.kind[0] in 'iufc':
            #print 'filling'
            #print df[k].name
            df[k].fillna(df[k].mean() if not math.isnan(df[k].mean()) else 0, inplace=True)
        elif df[k].dtype.kind == 'OS':
            #print 'strings'
            #print df[k].name
            df[k].fillna('None', inplace=True)

print 'Reading training data...'
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename='data/merged_train_2016_total.csv', target_dtype=np.float32, features_dtype=np.float32)


train_df = pd.read_csv('data/merged_train_2016_total.csv', parse_dates=['transactiondate'], nrows=50)
#print train_df
fillna_df(train_df)
feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32),
    #tf.contrib.layers.real_valued_column('poolsizesum'),
    #tf.contrib.layers.real_valued_column('bedroomcnt'),
    #tf.contrib.layers.real_valued_column('bathroomcnt'),
    #tf.contrib.layers.real_valued_column('yardbuildingsqft26'),
    #tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_hash_bucket('heatingorsystemtypeid',10), 10),


    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32),
    #tf.contrib.layers.real_valued_column('finishedsquarefeet13'),
    tf.contrib.layers.real_valued_column('totalinfo', dtype=tf.int64)
    #tf.contrib.layers.real_valued_column('buildingqualitytypeid')
]
#feature_columns.append(tf.contrib.layers.one_hot_column(feature_columns[3])),
#feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[1], boundaries=range(0, 10)))
#feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[2], boundaries=range(0, 10)))

model = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                      model_dir = './dnn-regressor-models_TEST',
                                      hidden_units = [4], dropout = 0, activation_fn = tf.nn.relu, optimizer=tf.train.AdagradOptimizer(learning_rate=.3), enable_centered_bias=True,
                                      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.
            CLASSES)
}
print 'Training...'
for _ in range(100):
    print 'iterations: %f' % (_+1)
    model.fit(input_fn=lambda: input_fn(train_df), steps=50)

#monitors=[tf.contrib.learn.monitors.ValidationMonitor(
#train_df['taxamount'],
#train_df['logerror'],
#every_n_steps=1,
#metrics=validation_metrics)]


print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016_total.csv', parse_dates=['transactiondate'])
#test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#    filename='data/merged_eval_2016_total.csv', target_dtype=np.float32, features_dtype=np.float32)
fillna_df(eval_df)
# eval_df['taxamount'].fillna(eval_df['taxamount'].mean(), inplace=True)
# eval_df['bathroomcnt'].fillna(eval_df['bathroomcnt'].mean(), inplace=True)
# eval_df['bedroomcnt'].fillna(eval_df['bedroomcnt'].mean(), inplace=True)
results = model.evaluate(input_fn=lambda: input_fn(train_df), steps=100)

print 'mean squared loss: %f' % results['loss']
print 'total loss: %f' % (results['loss'] * 45407)
print("--- %s seconds ---" % (time.time() - start_time))
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
