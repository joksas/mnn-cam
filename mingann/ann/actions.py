from . import architecture, data
import tensorflow as tf
import pickle
import os


DATASET = "mnist"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.h5")
INFO_PATH = os.path.join(MODELS_DIR, "info.pkl")
MAX_NUM_EPOCHS = 10


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    model = architecture.get_model(DATASET, is_training=True)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, save_weights_only=True),
            ]

    history = model.fit(
            data.load(DATASET, "training"),
            validation_data=data.load(DATASET, "validation"),
            verbose=2,
            epochs=MAX_NUM_EPOCHS,
            callbacks=callbacks,
            )

    info = {
            "history": history.history,
            "train_split_boundary": data.TRAIN_SPLIT_BOUNDARY,
            "batch_size": data.BATCH_SIZE,
            }

    with open(INFO_PATH, "wb") as handle:
        pickle.dump(info, handle)


def infer():
    model = architecture.get_model(DATASET, custom_weights_path=MODEL_PATH, is_training=False)
    model.evaluate(data.load(DATASET, "testing"))
