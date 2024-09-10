import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def load_dicom_images(patient_folder, image_size=(224, 224)):
    dicom_images = []
    image_count = 0

    for root, dirs, files in os.walk(patient_folder):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                try:
                    dicom_data = pydicom.dcmread(dicom_path)
                    dicom_image = dicom_data.pixel_array

                    # Img load empty or corrupt check
                    if dicom_image is None or dicom_image.size == 0:
                        print(f"Empty or invalid DICOM image: {dicom_path}")
                        continue

                    # normalize
                    dicom_image = cv2.resize(dicom_image, image_size)

                    # conv 2D to 3D grayscale, np stack
                    if len(dicom_image.shape) == 2:
                        dicom_image = np.stack((dicom_image,) * 3, axis=-1)

                    # post conv check
                    if dicom_image.shape != (image_size[0], image_size[1], 3):
                        print(f"Unexpected image shape: {dicom_image.shape} at {dicom_path}")
                        continue

                    dicom_images.append(dicom_image)
                    image_count += 1
                except Exception as e:
                    print(f"Error loading DICOM image {dicom_path}: {e}")
                    continue

    if image_count == 0:
        print(f"No valid DICOM images found in {patient_folder}")

    return np.array(dicom_images)

def load_data(folder, csv_file, limit_patients=None, image_size=(224, 224), max_images_per_patient=30):
    df = pd.read_csv(csv_file)
    images = []
    ejection_fractions = []

    if limit_patients:
        df = df[:limit_patients]

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        patient_folder = os.path.join(folder, str(int(row['Id'])), 'study')
        dicom_images = load_dicom_images(patient_folder, image_size)

        if dicom_images.size == 0:
            print(f"No images found for patient {row['Id']}, skipping...")
            continue

        # Handle different numbers of images by padding/truncating
        num_images = dicom_images.shape[0]
        if num_images < max_images_per_patient:
            # Pad with zeros to reach the fixed number of images
            padding = np.zeros((max_images_per_patient - num_images, image_size[0], image_size[1], 3))
            dicom_images = np.concatenate((dicom_images, padding), axis=0)
        elif num_images > max_images_per_patient:
            # Truncate to the fixed number of images
            dicom_images = dicom_images[:max_images_per_patient]

        # Compute Ejection Fraction (EF)
        edv = row['Diastole']
        esv = row['Systole']
        ef = ((edv - esv) / edv) * 100

        images.append(dicom_images)
        ejection_fractions.append(ef)

    return np.array(images), np.array(ejection_fractions)

# Data loading
train_images, train_efs = load_data('train/train', 'train.csv')
val_images, val_efs = load_data('validate/validate', 'validate.csv')
test_images, test_efs = load_data('test/test', 'solution.csv')

# Step 1: Flatten the image sequence
train_imgs = train_images.reshape((-1, 224, 224, 3))  # New shape: (number of frames, 224, 224, 3)
val_imgs = val_images.reshape((-1, 224, 224, 3))
test_imgs = test_images.reshape((-1, 224, 224, 3))

# Step 2: Repeat the ejection fraction (EF) values for each patient to match the number of frames
train_efs = np.repeat(train_efs, 30)  # New shape: (number of frames,)
val_efs = np.repeat(val_efs, 30)
test_efs = np.repeat(test_efs, 30)

# Split the data into training, validation, and test sets
train_imgs, val_imgs, train_ef, val_ef = train_test_split(train_imgs, train_efs, test_size=0.15, random_state=42)
test_imgs, _, test_ef, _ = train_test_split(test_imgs, test_efs, test_size=0.15, random_state=42)

# Define the ResNet model
def build_model(input_shape=(224, 224, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # dropout
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='linear')(x)  # Regression for EF prediction

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the ResNet layers except the last few layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
    return model

model = build_model()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Callbacks for early stopping and learning rate reduction
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Model training
history = model.fit(
    datagen.flow(train_imgs, train_ef, batch_size=32),
    epochs=100,
    validation_data=(val_imgs, val_ef),
    callbacks=callbacks
)

# Save the model
model.save('ejection_fraction_model.keras')

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(test_imgs, test_ef)
print(f"Test MAE: {test_mae}")

# Plot training history
plt.figure(figsize=(14, 5))

# Training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

# Training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()
