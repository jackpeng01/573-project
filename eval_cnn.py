import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# Load the trained model from disk.
model = tf.keras.models.load_model("cnn_classifier.h5")

# Print the model summary.
model.summary()

# Configuration for the test data.
img_size = (128, 128)
batch_size = 32

# Update if your test data is in a different folder.
data_dir = "data"

# Create a data generator for the test images (rescale images).
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load test images from the directory.
test_gen = test_datagen.flow_from_directory(
    data_dir,  # The directory containing your test data structured into subfolders.
    target_size=img_size,  # Resize images to 128x128.
    batch_size=batch_size,
    class_mode="binary",  # For binary classification; adjust if needed.
    shuffle=False,
)

# Evaluate the model on the test data.
score = model.evaluate(test_gen)

# Obtain predictions (assuming binary classification).
predictions = model.predict(test_gen)

# Convert probabilities to binary labels using 0.5 as the threshold.
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Extract true labels from the generator.
true_classes = test_gen.classes

# Compute evaluation metrics.
precision = precision_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
auc = roc_auc_score(true_classes, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC:", auc)

# Optional: Detailed classification report (includes support, etc.).
print(
    "\nClassification Report:\n", classification_report(true_classes, predicted_classes)
)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
print("Total test images:", test_gen.samples)
