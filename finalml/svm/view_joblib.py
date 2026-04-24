import joblib

# Load the saved files
model = joblib.load('svm_full_paper_style_outputs/svm_model.joblib')
scaler = joblib.load('svm_full_paper_style_outputs/scaler.joblib')
encoder = joblib.load('svm_full_paper_style_outputs/label_encoder.joblib')

print("\n✅ Files loaded successfully!\n")

print("Model details:")
print(model)

print("\nScaler details:")
print(scaler)

print("\nLabel Encoder classes:")
print(encoder.classes_)
