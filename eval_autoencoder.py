from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import matplotlib.pyplot as plt
import time  # For timing key operations
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import register_keras_serializable


# Define and register a custom mse function so that the model can find it.
@register_keras_serializable()
def mse(y_true, y_pred):
    """
    Custom Mean Squared Error function wrapped with the
    tensorflow.keras.losses.MeanSquaredError implementation.
    """
    return MeanSquaredError()(y_true, y_pred)


def load_test_data(data_dir, img_size, batch_size, color_mode="grayscale"):
    """
    Loads test data using ImageDataGenerator.
    Assumes that 'data_dir' contains subdirectories (e.g. 'defective' and 'non_defective').

    Parameters:
      - data_dir: Path to the directory.
      - img_size: Tuple specifying target image size.
      - batch_size: Batch size used for loading.
      - color_mode: Either 'grayscale' or 'rgb'. Set to match training data.

    Returns:
      - images: Numpy array of images.
      - labels: Corresponding labels.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",  # Assumes folder names imply binary labels (defective vs. non_defective)
        color_mode=color_mode,  # Change to 'rgb' if your model was trained on color images
        shuffle=False,
    )

    images_list, labels_list = [], []
    num_batches = len(test_gen)
    print(
        f"Total number of batches: {num_batches}"
    )  # Debug: Shows the total batches to process
    for i in range(num_batches):
        imgs, labels = next(test_gen)
        images_list.append(imgs)
        labels_list.append(labels)
        print(
            f"Processed batch {i + 1}/{num_batches}"
        )  # Debug: Indicates progress through the generator

    images = np.vstack(images_list)
    labels = np.concatenate(labels_list)
    return images, labels


def compute_reconstruction_error(autoencoder, images):
    """
    Computes the mean squared reconstruction error per image.
    Note: Added debug prints and a batch_size parameter to autoencoder.predict.
    """
    print("Starting model prediction on test images...")
    # Using a prediction batch size to avoid memory issues on large datasets.
    reconstructions = autoencoder.predict(images, batch_size=16)
    print("Prediction completed.")
    print(f"Reconstructions shape: {reconstructions.shape}")
    recon_errors = np.mean(np.square(images - reconstructions), axis=(1, 2, 3))
    return recon_errors


def main():
    # --- Configuration ---
    data_dir = "/Users/jackp/573/final/data"  # Root directory with subfolders 'defective' and 'non_defective'
    model_path = "autoencoder_wafer.h5"
    img_size = (128, 128)
    batch_size = 1  # Consider increasing this (e.g., 16 or 32) for faster processing if your system allows it
    threshold_multiplier = 2  # Threshold = mean + (threshold_multiplier * std)
    color_mode = (
        "grayscale"  # Change to 'rgb' if your model was trained with RGB images
    )

    # --- Load the trained autoencoder ---
    try:
        # Provide the custom object mapping for the 'mse' function.
        autoencoder = load_model(model_path, custom_objects={"mse": mse})
    except Exception as e:
        print("Error loading model with custom_objects:")
        print("Exception:", e)
        return  # Exit if the model cannot be loaded

    print(f"Loaded autoencoder model from {model_path}")

    # --- Load Test Data ---
    # Timer for loading data
    start_time = time.time()
    test_images, test_labels = load_test_data(
        data_dir, img_size, batch_size, color_mode=color_mode
    )
    data_loading_time = time.time() - start_time
    print(f"Data loading took {data_loading_time:.2f} seconds")
    print(f"Loaded {test_images.shape[0]} test images.")
    print(f"Test images shape: {test_images.shape}")

    # --- Compute Reconstruction Errors ---
    # Timer for model prediction
    start_time = time.time()
    recon_errors = compute_reconstruction_error(autoencoder, test_images)
    prediction_time = time.time() - start_time
    print(f"Model prediction took {prediction_time:.2f} seconds")

    # --- Compute Statistics and Threshold ---
    mean_err = np.mean(recon_errors)
    std_err = np.std(recon_errors)
    threshold = mean_err + threshold_multiplier * std_err

    print("\nReconstruction Error Statistics:")
    print(f"Mean Error: {mean_err:.6f}")
    print(f"Std. Dev.: {std_err:.6f}")
    print(f"Threshold (mean + {threshold_multiplier}*std): {threshold:.6f}")

    # --- Identify Anomalies ---
    anomaly_mask = recon_errors > threshold
    num_anomalies = np.sum(anomaly_mask)
    print(
        f"Detected anomalies (above threshold): {num_anomalies} out of {len(recon_errors)} images."
    )

    # --- Plot Histogram of Reconstruction Errors ---
    plt.figure(figsize=(10, 6))
    plt.hist(recon_errors, bins=50, alpha=0.75, label="Reconstruction Errors")
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.4f}"
    )
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of Images")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Convert anomaly mask to integer predictions for binary classification.
    predicted_labels = anomaly_mask.astype("int32")

    # --- Performance Metrics ---
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    auc = roc_auc_score(
        test_labels, recon_errors
    )  # Using reconstruction error as score

    print("Test Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("AUC:", auc)
    print(
        "\nClassification Report:\n",
        classification_report(test_labels, predicted_labels),
    )


if __name__ == "__main__":
    main()
