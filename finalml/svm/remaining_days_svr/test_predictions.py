import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load(r"C:\Users\Alekhya\Desktop\4-2 project\finalml\svm (3)\svm (2)\svm\remaining_days_svr\outputs\logreg_model.joblib")

# Test values for 1 day prediction
test_values = [0.13, 0.135, 0.139]

print("Testing VOC values for 1-day prediction:")
print("-" * 50)

for voc in test_values:
    X = pd.DataFrame({"VOC": [float(voc)]})
    pred = int(model.predict(X)[0])
    print(f"VOC: {voc:.3f} -> Predicted: {pred} days")

# Also check the actual decision boundaries
print("\n" + "=" * 50)
print("Checking decision boundaries around 0.13-0.14:")
print("-" * 50)

# Test fine-grained values around the threshold
for voc in np.arange(0.125, 0.145, 0.002):
    X = pd.DataFrame({"VOC": [float(voc)]})
    pred = int(model.predict(X)[0])
    print(f"VOC: {voc:.3f} -> Predicted: {pred} days")
