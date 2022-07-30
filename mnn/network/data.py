import logging

import tensorflow as tf
import tensorflow_datasets as tfds


def load(
    dataset: str, subset: str, batch_size: int, train_split_boundary: int = 80
) -> tf.data.Dataset:
    if subset == "training":
        split = f"train[:{train_split_boundary}%]"
    elif subset == "validation":
        split = f"train[{train_split_boundary}%:]"
    elif subset == "testing":
        split = "test"
    else:
        raise ValueError(f'Subset "{subset}" is not recognised!')

    ds = tfds.load(
        dataset,
        split=split,
        as_supervised=True,
        shuffle_files=True,
    )
    size = ds.cardinality().numpy()

    ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    if subset == "testing":
        ds = ds.batch(batch_size)
        ds = ds.cache()
    else:
        ds = ds.cache()
        ds = ds.shuffle(size)
        ds = ds.batch(batch_size)
        if dataset == "cifar10" and subset == "training":
            data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomTranslation(0.1, 0.1),
                    tf.keras.layers.RandomFlip("horizontal"),
                ]
            )
            ds = ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    ds = ds.prefetch(tf.data.AUTOTUNE)

    logging.info(f'Loaded dataset "{dataset}" ({subset}): {size} examples.')

    return ds


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label
