import tensorflow as tf
import pandas as pd

import models

print 'Reading evaluation data...'
eval_df = pd.read_csv('data/merged_eval_2016.csv', parse_dates=['transactiondate'], nrows=50)
models.fillna_df(eval_df)

model = models.dnn_regressor
results = model.evaluate(input_fn=lambda: models.input_fn(eval_df), steps=5)

print results
print model.get_variable_names()
print 'Logits Layer:'
for weight in model.get_variable_value('dnn/logits/weights').flatten():
    print '  %fx + %f' % (weight, model.get_variable_value('dnn/logits/biases')[0])
print model.get_variable_value('dnn/hiddenlayer_0/weights')
print model.get_variable_value('dnn/hiddenlayer_0/biases')

input_samples = eval_df.sample(n=10)
output_samples = list(model.predict(input_fn=lambda: models.input_fn(input_samples),
                                    outputs=None))
print 'Selected Results'
print '{:<20}{:<20}'.format('prediction', 'actual')
for k in range(len(output_samples)):
    print '{:<20f}{:<20f}'.format(output_samples[k], input_samples['logerror'].values[k])
