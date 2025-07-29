import tensorflow as tf
from tensorflow.keras import layers, models

print("Recreating model architecture...")

# Recreate model architecture (sama persis seperti di mesin.py)
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')
])

try:
    # Coba save model (kalau masih di memory)
    model.save("skincheck_model_tf.keras")
    print("Model berhasil disimpan!")
except:
    print("Model tidak ada di memory. Perlu training ulang atau load weights dari checkpoint.")