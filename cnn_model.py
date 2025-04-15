import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
img_size = (128, 128)    # Images will be resized to 128x128
batch_size = 32          # Batch size for training/test generators
data_dir = '/Users/jackp/573/final/data'       # Folder structured as:
                         # data/
                         # ├── defective/
                         # └── non_defective/

# --- Prepare Data Generators ---
# We define a single ImageDataGenerator with a validation_split.
# The 'training' subset will be used to train, and the 'validation' subset will act as our test set.
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

test_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# --- Define the CNN model ---
model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- Train the Model ---
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10
)

# --- Evaluate the Model on the Test Set ---
score = model.evaluate(test_gen)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# --- Save the Trained Model ---
model.save('cnn_classifier.h5')