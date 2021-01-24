import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kronos.evaluators.base import Evaluator

PATH = "/home/kevin/repos/kronos/kronos/logs/nov/clf.pkl"


class LinearProbe(Evaluator):
    def __init__(self, num_frames_per_seq=50):
        """Constructor.

        Args:
            num_frames_per_seq: The number of frames to sample in each sequence.
                This is so that we can generate a fixed-size dataset.

        Raises:
            ValueError: If the distance metric is invalid.
        """
        self.num_frames_per_seq = num_frames_per_seq

        self.clf = LogisticRegression(multi_class="multinomial", max_iter=5000)
        self.scaler = StandardScaler()

    def _evaluate(self, embs, labels, frames, fit=False):
        """Get pairwise nearest-neighbour frames.
        """
        X_train, y_train = [], []
        for X, y in zip(embs, labels):
            idxs = np.random.choice(
                np.arange(len(X)), replace=False, size=self.num_frames_per_seq
            )
            X_train.append(X[idxs])
            y_train.append(y[idxs])
        X_train = np.stack(X_train)
        y_train = np.stack(y_train)

        B, T, D = X_train.shape
        X_train = X_train.reshape(B * T, D)
        y_train = y_train.flatten()

        # X_train = self.scaler.fit_transform(X_train)

        if fit:
            self.clf = self.clf.fit(X_train, y_train)
            with open(PATH, "wb") as fp:
                pickle.dump(self.clf, fp)
        else:
            with open(PATH, "rb") as fp:
                self.clf = pickle.load(fp)

        return {
            "scalar": self.clf.score(X_train, y_train),
        }
