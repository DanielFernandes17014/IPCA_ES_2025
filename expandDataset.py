import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('cat_movement_labeled.csv')

# Number of times to replicate the data
replication_factor = 2  # Double the dataset size

# Create a new DataFrame to store expanded data
expanded_df = pd.DataFrame()

# Track the maximum SegmentID in the original dataset
max_segment_id = df['SegmentID'].max()

# Replicate rows with slight modifications
for i in range(replication_factor):
    # Add small random noise to X, Y, Z columns
    noisy_data = df.copy()
    noise = np.random.normal(0, 1, df[['X', 'Y', 'Z']].shape)  # Generate noise
    noisy_data[['X', 'Y', 'Z']] += noise  # Add noise

    # Round the noisy values to the nearest integer
    noisy_data[['X', 'Y', 'Z']] = noisy_data[['X', 'Y', 'Z']].round().astype(int)

    # Update SegmentID for the new replicated data
    noisy_data['SegmentID'] += (i + 1) * (max_segment_id + 1)

    # Append to the expanded DataFrame
    expanded_df = pd.concat([expanded_df, noisy_data], ignore_index=True)

# Combine the original and replicated data
final_df = pd.concat([df, expanded_df], ignore_index=True)

# Save the expanded dataset
final_df.to_csv('expanded_cat_movement_labeled_integers_fixed.csv', index=False)