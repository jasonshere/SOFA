import os
import numpy as np
import pandas as pd

# Ensure the directory exists
if not os.path.exists('data/diginetica'):
    os.makedirs('data/diginetica')

# Download the dataset
os.system("wget https://github.com/RecoHut-Datasets/diginetica/raw/main/train-item-views.csv -O data/diginetica/train-item-views.csv")

# Load dataset
ds_df = pd.read_csv('data/diginetica/train-item-views.csv', sep=';')

def process_ds(ds_df):
    """
    Processes the dataset by filtering sessions and items,
    sorting the dataset, and creating a timestamp column.
    """
    while True:
        # Filter out sessions with length <= 9
        session_size = ds_df.groupby('sessionId').size()
        filtered_session_ids = session_size[session_size > 9].index
        ds_df = ds_df[ds_df['sessionId'].isin(filtered_session_ids)]

        # Filter out items appearing less than 10 times
        item_size = ds_df.groupby('itemId').size()
        filtered_item_ids = item_size[item_size > 9].index
        ds_df = ds_df[ds_df['itemId'].isin(filtered_item_ids)]

        # Sort by eventdate, sessionId, and timeframe
        ds_df = ds_df.sort_values(['eventdate', 'sessionId', 'timeframe'])

        min_session_len = ds_df.groupby('sessionId').size().min()
        min_item_count = ds_df.groupby('itemId').size().min()

        if min_session_len > 9 and min_item_count > 9:
            break

    # Create timestamp column
    ds_df['timestamp'] = pd.to_datetime(ds_df['eventdate']).astype(int) // 10**9

    num_sessions = ds_df['sessionId'].nunique()
    num_items = ds_df['itemId'].nunique()

    print(f"The number of Sessions: {num_sessions}")
    print(f"The number of Items: {num_items}")

    return ds_df

# Process dataset
ds_df = process_ds(ds_df)

# Split the dataset into training and test sets
train_latest_date = ds_df['eventdate'].sort_values().unique()[-8]
training_df = ds_df[ds_df['eventdate'] <= train_latest_date]

# Get items appearing in the training set
items_in_training = training_df[training_df.groupby('sessionId').cumcount(ascending=False) > 0]['itemId'].unique()

# Create the test set and filter items not in the training set
test_df = ds_df[ds_df['eventdate'] > train_latest_date]
test_df = test_df[test_df['itemId'].isin(items_in_training)]

# Re-factorize sessionId, itemId, and cateId
ds_df = pd.concat([training_df, test_df])
ds_df['sessionId'] = pd.factorize(ds_df['sessionId'])[0]
ds_df['itemId'] = pd.factorize(ds_df['itemId'])[0]

# Recreate training and test sets
training_df = ds_df[ds_df['eventdate'] <= train_latest_date]
test_df = ds_df[ds_df['eventdate'] > train_latest_date]

# Save processed datasets
training_df.to_csv('data/diginetica/processed_training_ds.csv', index=False)
test_df.to_csv('data/diginetica/processed_test_ds.csv', index=False)

print("Processing completed. Training and test datasets saved.")
