import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from micromlgen import port


# 1. Load the CSV dataset
data = pd.read_csv('expanded_cat_movement_labeled_integers_fixed.csv')

# 2. Aggregate features by movement instance (segment)
grouped = data.groupby('SegmentID')

feature_list = []
label_list = []

for segment, group in grouped:
    features = []
    # For each sensor column, compute the mean and standard deviation.
    for col in ['X', 'Y', 'Z']:
        mean_val = group[col].mean()
        std_val = group[col].std()
        features.append(mean_val)
        features.append(std_val)
    feature_list.append(features)
    
    #Get all matching labels from segment
    label_list.append(group['Movement'].iloc[0])

# Convert lists into numpy arrays.
X_aggregated = np.array(feature_list)
y_labels = np.array(label_list)

print("Aggregated feature shape:", X_aggregated.shape)
print("Number of segments:", len(y_labels))

# 3. Encode string labels into numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Split the aggregated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_aggregated, y_encoded, test_size=0.2, random_state=42
)

# 5. Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators = 3,max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluate the model on the test set
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(port(rf_model))
with open('.\RF_classifier.h','w') as file:
    file.write(port(rf_model))
