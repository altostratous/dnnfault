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

LastValueQuantizer = quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = quantization.keras.quantizers.MovingAverageQuantizer


class DenseQuantizeConfig(quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      pass

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}


def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return quantization.keras.quantize_annotate_layer(layer)
    if isinstance(layer, tf.keras.layers.Dense):
        return quantization.keras.quantize_annotate_layer(layer, DenseQuantizeConfig())
    return layer


alexnet = tf.keras.models.clone_model(
    alexnet,
    clone_function=apply_quantization_to_dense,
)

with quantization.keras.quantize_scope(
  {'DenseQuantizeConfig': DenseQuantizeConfig}):
    alexnet = quantization.keras.quantize_apply(alexnet)


def process_images(image, label=None):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


batch_size = 16
(all_images, all_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
validation_images, validation_labels = all_images[:5000], all_labels[:5000]
# validation_images, validation_labels = all_images[:20], all_labels[:20]
train_images, train_labels = all_images[5000:], all_labels[5000:]
# train_images, train_labels = all_images[:20], all_labels[:20]
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
s = tf.data.Dataset.cardinality(test_ds)
test_ds = test_ds.shuffle(buffer_size=s).batch(batch_size, drop_remainder=True).map(process_images)
s = tf.data.Dataset.cardinality(train_ds)
train_ds = train_ds.shuffle(buffer_size=s).batch(batch_size, drop_remainder=True    ).map(process_images)

alexnet.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(lr=0.001),
    metrics=['accuracy'])


alexnet.fit(train_ds,
            epochs=50,
            validation_data=test_ds,
            validation_freq=1, callbacks=[tf.keras.callbacks.ModelCheckpoint(
                filepath='tmp/weights/quantized_alexnet/none',
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)])
