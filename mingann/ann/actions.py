import pickle
import os
from pathlib import Path
import tensorflow as tf
from . import architecture, data


class TrainingSetup:
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        train_split_boundary: int = 80,
        num_epochs: int = 1000,
        idx: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_split_boundary = train_split_boundary
        self.num_epochs = num_epochs
        self.idx = idx
        self.__training_data: tf.data.Dataset = None
        self.__validation_data: tf.data.Dataset = None
        print(self.model_path())

    def get_data(self, subset: str) -> tf.data.Dataset:
        if subset == "training":
            if self.__training_data is not None:
                return self.__training_data
        elif subset == "validation":
            if self.__validation_data is not None:
                return self.__validation_data

        dataset = data.load(self.dataset, subset, self.batch_size)

        if subset == "training":
            self.__training_data = dataset
        elif subset == "validation":
            self.__validation_data = dataset

        return dataset

    def dir(self):
        return os.path.join(
            Path(__file__).parent.parent.parent.absolute(), "models", f"network-{self.idx+1}"
        )

    def model_path(self):
        return os.path.join(self.dir(), "model.h5")

    def info_path(self):
        return os.path.join(self.dir(), "info.pkl")

    def info(self):
        return {
            "train_split_boundary": self.train_split_boundary,
            "batch_size": self.batch_size,
        }


class InferenceSetup:
    def __init__(self, training_setup: TrainingSetup, batch_size: int = 0) -> None:
        self.training = training_setup
        self.batch_size = batch_size
        if self.batch_size == 0:
            self.batch_size = self.training.batch_size
        self.__testing_data: tf.data.Dataset = None

    def get_data(self) -> tf.data.Dataset:
        if self.__testing_data is not None:
            return self.__testing_data

        self.__testing_data = data.load(self.training.dataset, "training", self.batch_size)

        return self.__testing_data


def train(setup: TrainingSetup):
    os.makedirs(setup.dir(), exist_ok=True)

    model = architecture.get_model(setup.dataset, is_memristive=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            setup.model_path(),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        setup.get_data("training"),
        validation_data=setup.get_data("validation"),
        verbose=2,
        epochs=setup.num_epochs,
        callbacks=callbacks,
    )

    info = {
        "history": history.history,
        **setup.info(),
    }

    with open(setup.info_path(), "wb") as handle:
        pickle.dump(info, handle)


def infer(setup: InferenceSetup):
    model = architecture.get_model(
        setup.training.dataset, custom_weights_path=setup.training.model_path(), is_memristive=False
    )
    model.evaluate(setup.get_data())

    model = architecture.get_model(
        setup.training.dataset, custom_weights_path=setup.training.model_path(), is_memristive=True
    )
    model.evaluate(setup.get_data())
