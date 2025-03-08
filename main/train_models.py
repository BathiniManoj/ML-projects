import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Classification Model (Traffic Density)
X_class = df[['Vehicle Count', 'Frame Area', 'Speed (m/s)']]
y_class = df['Density Category']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
print("Classification Accuracy:", accuracy_score(y_test_c, y_pred_c))

# Regression Model (Clearance Time Prediction)
X_reg = df[['Vehicle Count', 'Speed (m/s)']]
y_reg = df['Clearance Time (s)']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)
print("Regression MAE:", mean_absolute_error(y_test_r, y_pred_r))

# Save models
joblib.dump(clf, "density_classifier.pkl")
joblib.dump(reg, "clearance_time_predictor.pkl")
print("Models saved successfully!")
