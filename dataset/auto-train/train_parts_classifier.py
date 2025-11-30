import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# --- 1. Configuration ---

# Manually list any features you want to exclude from training.
# This is useful for experimenting. For example, let's exclude the last few angles.
EXCLUDED_FEATURES = [
    'angle_7','angle_8', 'angle_9', 'angle_10'
]

# --- 2. Data Preparation ---

# Load the dataset with advanced features
try:
    df = pd.read_csv('datas/datasets/shape_features/shape_features.csv')
except FileNotFoundError:
    print("Error: 'shape_features_advanced.csv' not found. Make sure the file is in the correct directory.")
    exit()

print("Dataset loaded successfully. Shape:", df.shape)

# Engineer a more robust aspect ratio from the minimum area rectangle dimensions
# This avoids division by zero and is rotation-invariant.
df['robust_aspect_ratio'] = np.maximum(df['min_rect_w'], df['min_rect_h']) / np.minimum(df['min_rect_w'], df['min_rect_h'])
# Fill any potential NaN/inf values that might result from zero division
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)


# Define the target label (y)
y = df['class_label']

# Define features (X) by removing non-feature columns AND the excluded list
non_feature_cols = ['class_label', 'source_image', 'mask_file']
all_features_df = df.drop(columns=non_feature_cols)

# Drop the manually excluded features
final_feature_cols = [col for col in all_features_df.columns if col not in EXCLUDED_FEATURES]
X = all_features_df[final_feature_cols]

print(f"\nTraining with {len(X.columns)} features.")

# Encode text labels into numbers (e.g., p1 -> 0, p2 -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nClass labels encoded:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"- {class_name}: {i}")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")


# --- 3. Model Training ---

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, class_weight='balanced')

# Train the Random Forest
print("\nTraining Random Forest Classifier...")
rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# Train the Support Vector Machine
print("\nTraining Support Vector Machine (SVM)...")
svm_model.fit(X_train, y_train)
print("SVM training complete.")


# --- 4. Model Evaluation ---

# Predictions from both models
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

print("\n--- Model Performance ---")

# Evaluate Random Forest
print("\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

# Evaluate SVM
print("\n--- Support Vector Machine (SVM) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))


# --- 5. Visualize Confusion Matrix ---

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Confusion Matrices', fontsize=16)

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0])
axes[0].set_title('Random Forest')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[1])
axes[1].set_title('Support Vector Machine (SVM)')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# --- 6. Save the Best Model and Supporting Files ---
best_model = rf_model # if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_svm) else svm_model
joblib.dump(best_model, "best_shape_classifier.joblib")
joblib.dump(scaler, "feature_scaler.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
joblib.dump(final_feature_cols, "feature_columns.joblib")


print("\nBest performing model and supporting files saved.")
print("These files are needed to make predictions on new data.")