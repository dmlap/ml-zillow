import tensorflow as tf
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

feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32),
    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32)
]

dnn_regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                              model_dir = './dnn-regressor-model',
                                              hidden_units = [4])
