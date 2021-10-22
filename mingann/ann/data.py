import tensorflow_datasets as tfds
import tensorflow as tf


TRAIN_SPLIT_BOUNDARY = 80
BATCH_SIZE = 64


def load(dataset, subset):
    if subset == "training":
        split = f"train[:{TRAIN_SPLIT_BOUNDARY}%]"
    elif subset == "validation":
        split = f"train[{TRAIN_SPLIT_BOUNDARY}%:]"
    elif subset == "testing":
        split = "test"
    else:
        raise ValueError(f"Subset \"{subset}\" is not recognised!")

    ds = tfds.load(
            dataset,
            split=split,
            as_supervised=True,
            shuffle_files=True,
            )
    size = ds.cardinality().numpy()

    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    if subset == "testing":
        ds = ds.batch(BATCH_SIZE)
        ds = ds.cache()
    else:
        ds = ds.cache()
        ds = ds.shuffle(size)
        ds = ds.batch(BATCH_SIZE)
        if dataset == "cifar10" and subset == "training":
            data_augmentation = tf.keras.Sequential([
              tf.keras.layers.RandomTranslation(0.1, 0.1),
              tf.keras.layers.RandomFlip("horizontal"),
            ])
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    print(f"Loaded dataset \"{dataset}\" ({subset}): {size} examples.")

    return ds


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

