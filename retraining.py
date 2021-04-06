import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization import quantization

alexnet = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                        input_shape=(227, 227, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
        return quantization.keras.quantize_annotate_layer(layer)
    return layer


alexnet = tf.keras.models.clone_model(
    alexnet,
    clone_function=apply_quantization_to_dense,
)

alexnet = quantization.keras.quantize_apply(alexnet)


def process_images(image, label=None):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


batch_size = 16
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
validation_images, validation_labels = train_images[:5000], train_labels[:5000]
# validation_images, validation_labels = train_images[:500], train_labels[:500]
train_images, train_labels = train_images[5000:], train_labels[5000:]
# train_images, train_labels = train_images[:500], train_labels[:500]
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
s = tf.data.Dataset.cardinality(test_ds)
test_ds = test_ds.shuffle(buffer_size=s).batch(batch_size).map(process_images)
s = tf.data.Dataset.cardinality(train_ds)
train_ds = train_ds.shuffle(buffer_size=s).batch(batch_size).map(process_images)

alexnet.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(lr=0.001),
    metrics=['accuracy'])
alexnet.run_eagerly = True

alexnet.fit(train_ds,
            epochs=50,
            validation_data=test_ds,
            validation_freq=1, callbacks=[tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp/weights/quantized_alexnet/none',
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)])
