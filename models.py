import tensorflow as tf
import math
import numpy as np

def input_fn(df):
    columns = {
        k: tf.constant(df[k].values)
        for k in [col.name for col in feature_columns] if not k.endswith('_bucketized')
    }
    print 'variance:'
    print df['logerror'].var()
    output = tf.constant(df['logerror'].values, dtype=tf.float64)

    print 'columns, output'
    print columns, output
    return columns, output

def fillna_df(df):
    for k in df:
        if df[k].dtype.kind in 'iufc' and df[k].name != 'logerror':
            df[k].fillna(df[k].mean() if not math.isnan(df[k].mean()) else 0, inplace=True)
            df[k]=(df[k]-df[k].mean())/df[k].std()

feature_columns = [
    tf.contrib.layers.real_valued_column('taxamount', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('yearbuilt', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('totalinfo', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('bedroomcnt', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('calculatedbathnbr', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('calculatedfinishedsquarefeet', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('fullbathcnt', dtype=tf.float32, dimension=1),
    #tf.contrib.layers.real_valued_column('2error', dtype=tf.float32, dimension=1), #test feature: x=2/logerror
    tf.contrib.layers.real_valued_column('basementsqft', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('finishedsquarefeet12', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('finishedsquarefeet13', dtype=tf.float32, dimension=1),
    tf.contrib.layers.real_valued_column('yardbuildingsqft26', dtype=tf.float32, dimension=1),


]
#feature_columns.append(tf.contrib.layers.bucketized_column(feature_columns[1], boundaries=range(0, 10)))
dnn_regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                              model_dir = './dnn-regressor-model_outl',
                                              hidden_units = [256,256,256,256],
                                              activation_fn = tf.nn.relu,
                                              dropout = .5,
                                              enable_centered_bias = True,
                                              label_dimension = 1,
                                              optimizer= tf.train.AdadeltaOptimizer(
                                              learning_rate=1,
                                              rho=0.99
                                              )
                                              )
