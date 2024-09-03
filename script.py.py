import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


# random seed
np.random.seed(42)
tf.random.set_seed(42)

def load_nii_file(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def preprocess_image(image_data, target_size=(256, 256)):
    from skimage.transform import resize
    if len(image_data.shape) == 3:
        middle_slice = image_data[:, :, image_data.shape[2] // 2]
    elif len(image_data.shape) == 4:
        middle_slice = image_data[:, :, image_data.shape[2] // 2, image_data.shape[3] // 2]
    else:
        middle_slice = image_data
    if len(middle_slice.shape) > 2:
        middle_slice = np.squeeze(middle_slice)
    image_resized = resize(middle_slice, target_size, mode='constant', preserve_range=True)
    image = image_resized.astype(np.float32)
    image /= np.max(image)  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)
    return image


def build_resnet(input_shape):
    # Adjust the input shape for a single-channel input
    inputs = Input(shape=input_shape)
    
    # Manually adjust the first convolutional layer to accept 1 channel instead of 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
    
    # Use ResNet50 architecture without the top layer (include_top=False)
    base_model = ResNet50(weights=None, include_top=False, input_tensor=x)
    
    # Add custom layers for the prediction task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def load_dataset(data_dir):
    images = []
    ejection_fractions = []
    
    patients = [p for p in os.listdir(data_dir) if p.startswith('patient') and os.path.isdir(os.path.join(data_dir, p))]
    
    for patient in patients:
        patient_dir = os.path.join(data_dir, patient)
        image_filename = f'{patient}_4d.nii'
        info_filename = os.path.join(patient_dir, 'info.cfg')
        
        # Load and preprocess image data
        image_path = os.path.join(patient_dir, image_filename)
        image_data = load_nii_file(image_path)
        image = preprocess_image(image_data)
        images.append(image)
        
        with open(info_filename, 'r') as f:
            edv, esv = None, None
            for line in f:
                if 'ED:' in line:
                    edv = int(line.split(':')[1].strip())
                elif 'ES:' in line:
                    esv = int(line.split(':')[1].strip())
                
            if edv is not None and esv is not None:
                ef = (edv - esv) / edv * 100  # Calculate EF
                ejection_fractions.append(ef)
            else:
                print(f"[Warning] EDV/ESV values missing for {patient}. Skipping EF calculation.")

    images = np.array(images)
    ejection_fractions = np.array(ejection_fractions)
    return images, ejection_fractions

training_folder = 'Training'

# Load training data
print("Loading training data...")
images, ejection_fractions = load_dataset(training_folder)

# Split into training and validation sets (701515)
X_train, X_val, y_train, y_val = train_test_split(images, ejection_fractions, test_size=0.2, random_state=42)

# Define model parameters
input_shape = (256, 256, 1)

model = build_resnet(input_shape)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("ef_cnn_model.keras", save_best_only=True, monitor='val_loss'),
    # tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss', restore_best_weights=True)
]

# Train the model (Hyperparams)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=11,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()

# Save the final model
model.save('ef_cnn_final_model.keras')

# Optional: Evaluate on validation set
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation MAE: {val_mae}")
