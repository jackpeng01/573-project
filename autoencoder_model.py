from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
img_size = (128, 128)  # Target image size
batch_size = 32  # Batch size for training/validation
epochs = 10  # Number of training epochs
data_dir = "/Users/jackp/573/final/data"  # Directory with subfolders 'defective' and 'non_defective'

# --- Prepare Data Generators for Training and Validation ---
# For training the autoencoder, we use only the 'non_defective' images.
train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="input",  # No labels needed for an autoencoder
    subset="training",
    classes=[
        "non_defective"
    ],  # Use only non-defective images for training the autoencoder.
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="input",
    subset="validation",
    classes=["non_defective"],
)

# Debug info: number of samples.
print("Training samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)
if train_gen.samples == 0 or val_gen.samples == 0:
    raise ValueError(
        "No samples found in one of the generators. Check that your 'non_defective' folder is populated."
    )

# Calculate steps explicitly.
steps_per_epoch = (train_gen.samples + batch_size - 1) // batch_size
validation_steps = (val_gen.samples + batch_size - 1) // batch_size

# --- Build the Convolutional Autoencoder ---
input_img = Input(shape=(*img_size, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# --- Train the Autoencoder ---
history = autoencoder.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=epochs,
    # use_multiprocessing=False,  # Disable multiprocessing
    # workers=1                   # Use a single worker
)

# Save the trained autoencoder
autoencoder.save("autoencoder_wafer.h5")

# --- Test the Autoencoder ---
# For testing, load images from both 'defective' and 'non_defective' folders.
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=1,  # Process one image at a time to compute error per image.
    class_mode="binary",  # Get class labels (defective or non_defective) if needed.
    shuffle=False,  # Keep consistent order.
)

# Collect test images and labels.
test_images = []
test_labels = []
for i in range(len(test_gen)):
    batch = next(test_gen)
    if isinstance(batch, tuple):
        images, labels = batch
    else:
        images = batch
        labels = None
    test_images.append(images)
    if labels is not None:
        test_labels.append(labels)
test_images = np.vstack(test_images)
test_labels = np.concatenate(test_labels) if test_labels else None

# Reconstruct images using the autoencoder.
reconstructions = autoencoder.predict(test_images)

# Compute reconstruction error for each image.
recon_errors = np.mean(np.square(test_images - reconstructions), axis=(1, 2, 3))

print("Reconstruction Error Statistics:")
print("Mean Error:", np.mean(recon_errors))
print("Std. Dev.:", np.std(recon_errors))

# --- Display a Few Examples ---
num_samples_to_display = 5
fig, axes = plt.subplots(
    num_samples_to_display, 2, figsize=(8, 4 * num_samples_to_display)
)
for i in range(num_samples_to_display):
    axes[i, 0].imshow(test_images[i])
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(reconstructions[i])
    axes[i, 1].set_title(f"Reconstructed\nError: {recon_errors[i]:.4f}")
    axes[i, 1].axis("off")
plt.tight_layout()
plt.show()
