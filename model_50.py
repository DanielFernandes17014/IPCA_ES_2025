import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import m2cgen as m2c  

# 1. Load the CSV dataset
data = pd.read_csv('cat_movement_labeled.csv')

# 2. Prepare features and labels
X = data[['X', 'Y', 'Z']].values
y = data['Movement'].values

# 3. Encode string labels into numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluate the model on the test set
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the model as a pickle file
joblib.dump(rf_model, 'movement_rf_model.pkl')
print("Random Forest model saved as movement_rf_model.pkl")

# 8. Export the model as C code using m2cgen
c_code = m2c.export_to_c(rf_model)
with open("rf_model.c", "w") as f:
    f.write(c_code)
print("C code for the Random Forest model saved as rf_model.c")
