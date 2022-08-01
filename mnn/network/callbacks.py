import logging

from tensorflow import keras


class TrainingLogging(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"Epoch {epoch+1}/{self.params['epochs']}\tloss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}"
        )
