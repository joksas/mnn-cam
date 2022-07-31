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


class TrainingConfig:
    def __init__(
        self,
        batch_size: int,
        num_repeats: int,
        train_split_boundary: int = 80,
        num_epochs: int = 1000,
        idx: int = 0,
    ) -> None:
        self.dataset: str = None
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.train_split_boundary = train_split_boundary
        self.num_epochs = num_epochs
        self.__idx = idx

    @staticmethod
    def models_dir():
        return os.path.join(Path(__file__).parent.parent.parent.absolute(), "models")

    def dir(self):
        return os.path.join(self.models_dir(), self.dataset, f"network-{self.__idx+1}")

    def model_path(self):
        return os.path.join(self.dir(), "model.h5")

    def info_path(self):
        return os.path.join(self.dir(), "info.pkl")

    def info(self):
        return {
            "dataset": self.dataset,
            "train_split_boundary": self.train_split_boundary,
            "batch_size": self.batch_size,
        }

    def next_iteration(self):
        self.__idx += 1

    def reset(self):
        self.__idx = 0

    def get_idx(self):
        return self.__idx


class InferenceConfig:
    def __init__(self, num_repeats: int, batch_size: int = 1000) -> None:
        self.num_repeats = num_repeats
        self.batch_size = batch_size


class SimulationConfig:
    def __init__(
        self,
        dataset: str,
        training_config: TrainingConfig,
        inference_config: InferenceConfig = None,
        k_V: float = 0.5,
        G_off: float = None,
        G_on: float = None,
        mapping_rule: str = "default",
        nonidealities: list[crossbar.nonidealities.Nonideality] = None,
    ):
        training_config.dataset = dataset
        self.__training = training_config
        self.__inference = inference_config
        self.__nonidealities = nonidealities
        self.__k_V = k_V
        self.__G_off = G_off
        self.__G_on = G_on
        self.__mapping_rule = mapping_rule
        self.__nonidealities = nonidealities
        self.__data: dict = {}

    def __get_data(self, subset: str) -> tf.data.Dataset:
        if subset in self.__data:
            return self.__data[subset]

        if subset == "testing":
            batch_size = self.__inference.batch_size
        else:
            batch_size = self.__training.batch_size

        self.__data[subset] = data.load(self.__training.dataset, subset, batch_size)

        return self.__data[subset]

    def __train_iteration(self):
        if os.path.isdir(self.__training.dir()):
            logging.warning(
                f'Training directory "{self.__training.dir()}" already exists. Skipping...'
            )
            return
        logging.info(
            "Starting to train network %d/%d.",
            self.__training.get_idx() + 1,
            self.__training.num_repeats,
        )

        os.makedirs(self.__training.dir(), exist_ok=True)

        model = architecture.get_model(self.__training.dataset)

        custom_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.__training.model_path(),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
            ),
            callbacks.TrainingLogging(),
        ]

        history = model.fit(
            self.__get_data("training"),
            validation_data=self.__get_data("validation"),
            verbose=0,
            epochs=self.__training.num_epochs,
            callbacks=custom_callbacks,
        )

        info = {
            "history": history.history,
            **self.__training.info(),
        }

        with open(self.__training.info_path(), "wb") as handle:
            pickle.dump(info, handle)

    def __infer_iteration(self):
        scores = [[], []]
        config = {
            "k_V": self.__k_V,
            "G_off": self.__G_off,
            "G_on": self.__G_on,
            "mapping_rule": self.__mapping_rule,
            "nonidealities": self.__nonidealities,
            "power_path": self.__test_power_temp_path(),
        }
        for memristive_config in [None, config]:
            model = architecture.get_model(
                self.__training.dataset,
                custom_weights_path=self.__training.model_path(),
                memristive_config=memristive_config,
            )
            score = model.evaluate(self.__get_data("testing"), verbose=0)
            scores[0].append(score[0])
            scores[1].append(score[1])

        return scores

    def __test_loss_path(self):
        return os.path.join(self.__training.models_dir(), "loss")

    def __test_accuracy_path(self):
        return os.path.join(self.__training.models_dir(), "accuracy")

    def __test_power_temp_path(self):
        return os.path.join(self.__training.models_dir(), "power-temp.csv")

    def __test_power_path(self):
        return os.path.join(self.__training.models_dir(), "power")

    def __load_temp_power(self):
        power = np.loadtxt(self.__test_power_temp_path())
        # Two synaptic layers.
        return 2 * np.mean(power)

    def __delete_temp_power(self):
        return os.remove(self.__test_power_temp_path())

    def test_loss(self):
        with open(self.__test_loss_path(), "rb") as file:
            return np.load(file)

    def test_accuracy(self):
        with open(self.__test_accuracy_path(), "rb") as file:
            return np.load(file)

    def test_power(self):
        with open(self.__test_power_path(), "rb") as file:
            return np.load(file)

    def train(self):
        self.__training.reset()
        for _ in range(self.__training.num_repeats):
            self.__train_iteration()
            self.__training.next_iteration()

        self.__training.reset()

    def infer(self):
        loss = np.zeros((self.__training.num_repeats, self.__inference.num_repeats, 2))
        accuracy = np.zeros((self.__training.num_repeats, self.__inference.num_repeats, 2))
        power = np.zeros((self.__training.num_repeats, self.__inference.num_repeats))
        self.__training.reset()
        for training_idx in range(self.__training.num_repeats):
            for inference_idx in range(self.__inference.num_repeats):
                scores = self.__infer_iteration()
                loss[training_idx, inference_idx, :] = scores[0]
                accuracy[training_idx, inference_idx, :] = scores[1]
                power[training_idx, inference_idx] = self.__load_temp_power()
                self.__delete_temp_power()
            self.__training.next_iteration()

        self.__training.reset()

        utils.save_numpy(self.__test_loss_path(), loss)
        utils.save_numpy(self.__test_accuracy_path(), accuracy)
        utils.save_numpy(self.__test_power_path(), power)
