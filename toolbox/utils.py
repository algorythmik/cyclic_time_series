import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.base import TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
import joblib
import tensorflow as tf
import tslearn
import tslearn.clustering

import matplotlib.pyplot as plt

data_dir = os.environ.get('DATA_DIR')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    classes = [classes[l] for l in unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def dict_to_time_series(data_dict):

    return to_time_series_dataset([list(zip(
        data_dict['force'],
        data_dict['position']
    ))])


def to_time_series_dataset(dataset):
    """Transforms a time series dataset so that it has the following format:
    (no_time_series, no_time_samples, no_features)

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    Returns
    -------
    numpy.ndarray of shape
        (no_time_series, no_time_samples, no_features)
    """
    assert len(dataset) != 0, 'dataset is empty'

    try:
        np.array(dataset, dtype=np.float)
    except ValueError:
        raise AssertionError('All elements must have the same length.')

    if np.array(dataset[0]).ndim == 0:
        dataset = [dataset]

    if np.array(dataset[0]).ndim == 1:
        no_time_samples = len(dataset[0])
        no_features = 1
    else:
        no_time_samples, no_features = np.array(dataset[0]).shape

    return np.array(dataset, dtype=np.float).reshape(
        len(dataset),
        no_time_samples,
        no_features)


def to_dataset(dataset):
    """Transforms a time series dataset so that it has the following format:
    (no_time_series, no_time_samples, no_features) where no_time_samples
    for different time sereies can be different.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    Returns
    -------
    list of np.arrays
        (no_time_series, no_time_samples, no_features)
    """
    assert len(dataset) != 0, 'dataset is empty'

    if np.array(dataset[0]).ndim == 0:
        dataset = [[d] for d in dataset]

    if np.array(dataset[0]).ndim == 1:
        no_features = 1
        dataset = [[[d] for d in data] for data in dataset]
    else:
        no_features = len(dataset[0][0])

    for data in dataset:
        try:
            array = np.array(data, dtype=float)
        except ValueError:
            raise AssertionError(
                "All samples must have the same number of features!")
        assert array.shape[-1] == no_features,\
            'All series must have the same no features!'

    return dataset


class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    no_output_samples : int
        Size of the output time series.
    """
    def __init__(self, no_output_samples):
        self._sz = no_output_samples

    def fit(self, X, y=None, **kwargs):
        return self

    def _interp(self, x):
        return np.interp(
            np.linspace(0, 1, self._sz),
            np.linspace(0, 1, len(x)),
            x)

    def transform(self, X, **kwargs):
        X_ = to_dataset(X)
        res = [np.apply_along_axis(self._interp, 0, x) for x in X_]
        return to_time_series_dataset(res)


class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp.
    standard deviation) in each dimension. The mean and std can either be
    constant (one value per feature over all times) or time varying (one value
    per time step per feature).

    Parameters
    ----------
    kind: str (one of 'constant', or 'time-varying')
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.
    """
    def __init__(self, kind='time-varying', mu=0., std=1.):
        assert kind in ['time-varying', 'constant'],\
            'axis should be one of time-varying or constant'
        self._axis = (1, 0) if kind == 'constant' else 0
        self.mu_ = mu
        self.std_ = std

    def fit(self, X, y=None, **kwargs):
        X_ = to_time_series_dataset(X)
        self.mean_t = np.mean(X_, axis=self._axis)
        self.std_t = np.std(X_, axis=self._axis)
        self.std_t[self.std_t == 0.] = 1.

        return self

    def transform(self, X, **kwargs):
        """Fit to data, then transform it.
        Parameters
        ----------
        X
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = to_time_series_dataset(X)
        X_ = (X_ - self.mean_t) * self.std_ / self.std_t + self.mu_

        return X_


def load_dataset(raw=False):
    """Loads and cleans the dataset in a dataframe, for raw data set raw to True.
    """
    label_map = {
        'Normal': 0,
        'PPU low': 1,
        'Flap broken': 2,
        'Flap missing': 3,
        'Pen missing in B': 0,
        'Pen missing in A': 0,
    }
    df = pd.read_pickle(os.path.join(
        data_dir, 'dataframes', '20191031', 'df.pkl'))
    df['label'] = df['operation_mode'].apply(lambda x: label_map[x])

    if raw:
        return df

    bad_label_run_ids = []
    groups_with_wrong_labels = [
        'K08', 'K09', 'K10', 'K11', 'K12', 'K13',
        'K24', 'K25', 'K26', 'K27', 'K28',
        'K29']

    for (name, nest), group in df.groupby(['group', 'nest']):
        group_idx = group.set_index('run_id')
        sorted_idx = sorted(group_idx.index.unique())
        if name in groups_with_wrong_labels:
            bad_label_run_ids.extend(sorted_idx[:2])
            bad_label_run_ids.extend(sorted_idx[-2:])
        if name == 'K08':
            bad_label_run_ids.extend(sorted_idx[:21])

    return df[~df['run_id'].isin(bad_label_run_ids)]


def split_df(df, feature_columns=['force', 'position']):
    labels = []
    features = []
    for id_, group in df.groupby('run_id'):
        features.append(group[feature_columns].values.tolist())
        labels.append(group['label'].iloc[0])

    return features, labels


class SklearnKerasPipe(_BaseComposition):
    """
    A pipeline contatining a sequence of transformers and a
    keras compiled model as the last step.
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y, **fit_params):

        X_val, y_val = fit_params.pop('validation_data')

        for step in self.steps[:-1]:
            step.fit(X, y)
            X = step.transform(X)
            X_val = step.transform(X_val)

        history = self.steps[-1].fit(
            X, y, validation_data=(X_val, y_val), **fit_params)

        return history

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        for step in self.steps[:-1]:
            X = step.transform(X)

        return self.steps[-1].predict(X)

    def score(self, X, y, **params):

        for step in self.steps[:-1]:
            X = step.transform(X)

        return self.steps[-1].evaluate(X, y, **params)

    def pickle(self, path):
        os.makedirs(path, exist_ok=True)
        transformers = self.steps[:-1]
        keras_model = self.steps[-1]

        joblib.dump(
            transformers,
            os.path.join(path, 'transformers.joblib')
        )
        keras_model.save(os.path.join(path, 'model.h5'))

    @classmethod
    def load_pipe(cls, path):
        transformers = joblib.load(os.path.join(path, 'transformers.joblib'))
        model = tf.keras.models.load_model(os.path.join(path, 'model.h5'))
        transformers.append(model)
        return cls(transformers)


class CutLastPart(TransformerMixin):
    """Cuts the last part of the curves.
    """

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        res = []
        for x in X:
            idx = np.argmax(np.array(x)[:, 0])
            res.append(x[:idx])

        return res


class KmeansAnomaly(tslearn.clustering.TimeSeriesKMeans):

    def __init__(self, **kwargs):
        self._threshold = kwargs.pop('threshold')
        super().__init__(**kwargs)

    def fit(self, x, y=None):
        super().fit(x)
        return self

    def predict_proba(self, x):
        return tslearn.metrics.cdist_dtw(
            x,
            self.cluster_centers_,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=25,
            n_jobs=-1).min(axis=1)

    def predict(self, x):
        return (self.predict_proba(x) < self._threshold).astype(int)
