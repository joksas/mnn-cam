import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from mnn import crossbar

from . import architecture, callbacks, data, utils

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.WARNING)

_GEN_DIR = os.path.join(Path(__file__).parent.parent.parent.absolute(), "_gen")


class TrainingConfig:
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        num_repeats: int,
        train_split_boundary: int = 80,
        num_epochs: int = 1000,
    ) -> None:
        self.dataset: str = dataset
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.train_split_boundary = train_split_boundary
        self.num_epochs = num_epochs
        self.__idx = 0
        self.__data: dict = {}

    def dir(self):
        return os.path.join(_GEN_DIR, self.dataset, "training")

    def network_dir(self):
        return os.path.join(self.dir(), str(self.__idx + 1))

    def model_path(self):
        return os.path.join(self.network_dir(), "model.h5")

    def info_path(self):
        return os.path.join(self.network_dir(), "info.pkl")

    def info(self):
        return {
            "dataset": self.dataset,
            "train_split_boundary": self.train_split_boundary,
            "batch_size": self.batch_size,
        }

    def reset(self):
        self.__idx = 0

    def get_idx(self):
        return self.__idx

    def get_data(self, subset: str, batch_size: int = None) -> tf.data.Dataset:
        if subset in self.__data:
            return self.__data[subset]

        if batch_size is None:
            batch_size = self.batch_size

        self.__data[subset] = data.load(self.dataset, subset, batch_size)

        return self.__data[subset]

    def __run_iteration(self):
        if os.path.isdir(self.network_dir()):
            logging.warning(
                f'Training directory "{self.network_dir()}" already exists. Skipping...'
            )
            return
        logging.info(
            "Starting to train network %d/%d.",
            self.__idx + 1,
            self.num_repeats,
        )

        os.makedirs(self.network_dir(), exist_ok=True)

        model = architecture.get_model(self.dataset)

        custom_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path(),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
            ),
            callbacks.TrainingLogging(),
        ]

        history = model.fit(
            self.get_data("training"),
            validation_data=self.get_data("validation"),
            verbose=0,
            epochs=self.num_epochs,
            callbacks=custom_callbacks,
        )

        info = {
            "history": history.history,
            **self.info(),
        }

        with open(self.info_path(), "wb") as handle:
            pickle.dump(info, handle)

    def next_iteration(self):
        self.__idx += 1

    def run(self):
        self.reset()
        for _ in range(self.num_repeats):
            self.__run_iteration()
            self.next_iteration()

        self.reset()


class InferenceConfig:
    def __init__(
        self,
        training_config: TrainingConfig,
        nonidealities: list[crossbar.nonidealities.Nonideality],
        num_repeats: int,
        batch_size: int = 1000,
        k_V: float = 0.5,
        G_off: float = None,
        G_on: float = None,
        mapping_rule: str = "default",
    ) -> None:
        self.__training = training_config
        self.__nonidealities = nonidealities
        self.__num_repeats = num_repeats
        self.__batch_size = batch_size
        self.__k_V = k_V
        self.__G_off = G_off
        self.__G_on = G_on
        self.__mapping_rule = mapping_rule

    def nonidealities_label(self):
        if len(self.__nonidealities) == 0:
            return "ideal"
        return "+".join(nonideality.label() for nonideality in self.__nonidealities)

    def dir(self):
        return os.path.join(
            _GEN_DIR, self.__training.dataset, "inference", self.nonidealities_label()
        )

    def __run_iteration(self):
        config = {
            "k_V": self.__k_V,
            "G_off": self.__G_off,
            "G_on": self.__G_on,
            "mapping_rule": self.__mapping_rule,
            "nonidealities": self.__nonidealities,
            "power_path": self.__power_temp_path(),
        }
        model = architecture.get_model(
            self.__training.dataset,
            custom_weights_path=self.__training.model_path(),
            memristive_config=config,
        )
        score = model.evaluate(self.__training.get_data("testing"), verbose=0)

        return score

    def __loss_path(self):
        return os.path.join(self.dir(), "loss")

    def __accuracy_path(self):
        return os.path.join(self.dir(), "accuracy")

    def __power_temp_path(self):
        return os.path.join(self.dir(), "power-temp.csv")

    def __power_path(self):
        return os.path.join(self.dir(), "power")

    def __load_temp_power(self):
        power = np.loadtxt(self.__power_temp_path())
        # Two synaptic layers.
        return 2 * np.mean(power)

    def __delete_temp_power(self):
        return os.remove(self.__power_temp_path())

    def loss(self):
        with open(self.__loss_path(), "rb") as file:
            return np.load(file)

    def accuracy(self):
        with open(self.__accuracy_path(), "rb") as file:
            return np.load(file)

    def power(self):
        with open(self.__power_path(), "rb") as file:
            return np.load(file)

    def run(self):
        os.makedirs(self.dir(), exist_ok=True)

        loss = np.zeros((self.__training.num_repeats, self.__num_repeats))
        accuracy = np.zeros((self.__training.num_repeats, self.__num_repeats))
        power = np.zeros((self.__training.num_repeats, self.__num_repeats))
        self.__training.reset()
        for training_idx in range(self.__training.num_repeats):
            for inference_idx in range(self.__num_repeats):
                scores = self.__run_iteration()
                loss[training_idx, inference_idx] = scores[0]
                accuracy[training_idx, inference_idx] = scores[1]
                power[training_idx, inference_idx] = self.__load_temp_power()
                self.__delete_temp_power()
            self.__training.next_iteration()

        self.__training.reset()

        utils.save_numpy(self.__loss_path(), loss)
        utils.save_numpy(self.__accuracy_path(), accuracy)
        utils.save_numpy(self.__power_path(), power)
