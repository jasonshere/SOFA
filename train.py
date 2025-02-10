import pandas as pd
import tensorflow as tf
from data.data_generator import prepare_datasets
from metrics import IED, SessionIED
from model import SOFA


training_df = pd.read_csv('data/diginetica/processed_training_ds.csv')
training_df['itemId'] += 1
test_df = pd.read_csv('data/diginetica/processed_test_ds.csv')
test_df['itemId'] += 1
ds_df = pd.concat([training_df, test_df])
num_items = ds_df.itemId.unique().shape[0]
session_max_length = ds_df.groupby('sessionId').size().max() - 1
training_sessions = training_df.groupby('sessionId').itemId.apply(list).values
test_sessions = test_df.groupby('sessionId').itemId.apply(list).values


training_ds, test_ds = prepare_datasets(session_max_length, training_sessions, test_sessions)

id = 1
test_id = 1

batch_size = 8
metrics = [
    tf.keras.metrics.TopKCategoricalAccuracy(k=15, name='HR@15'),
    tf.keras.metrics.TopKCategoricalAccuracy(k=20, name='HR@20'),

    # IED
    IED(num_items, 15, name='IED@15'),
    IED(num_items, 20, name='IED@20'),

    # Session IED
    # SessionIED(num_items, 15, name='SIED@15'),
    # SessionIED(num_items, 20, name='SIED@20'),
]

model = SOFA(num_items,
                emb_dim=200,
                session_max_length=session_max_length,
                soda_steps=5,
                lambda_=10,
                epi=1,
                train_fair=True,
                threshold=1.0,
                fairness_steps=1,
                soup_lr=0.0002,
                soda_lr=0.1,
                soup_drop_out=0.,
                soda_drop_out=0.,
                fair_kernel_size=2,
                fair_tcn_emb_dim=250,
                use_drsn=True,
                soda_drsn=True,
                use_position_emb=True,
                tcn_emb_dim=10)
model.compile(optimizer='adam', loss=None, run_eagerly=False, metrics=metrics)
model.fit(training_ds.batch(batch_size, num_parallel_calls=5), epochs=100, validation_freq=1, validation_data=test_ds.batch(batch_size), callbacks=[])