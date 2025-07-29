import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# Deteksi GPU
print("Using device:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# Path dataset
DATASET_PATH = "dataSetDigistar"  # ubah ke path lokal

# Load dataset dengan split 80% training dan 20% validation
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Normalisasi (opsional, tapi disarankan)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Ambil model pretrained dari Keras Applications
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze layer pretrained


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')  # 2 kelas
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
EPOCHS = 5
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Simpan model
model.save_weights("skincheck_model_tf.weights.h5")
print("Training selesai dan model disimpan!")
