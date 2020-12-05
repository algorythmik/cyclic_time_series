import os

from toolbox.utils import SklearnKerasPipe
import joblib

saved_model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'saved_models'
)


class Classifier:

    _classes = [
        'Normal',
        'low_ppu',
        'flap_broken',
        'flap_missing',
    ]

    _classifer_path = os.path.join(saved_model_path, 'multi_kernel')

    def __init__(self):
        self._model = SklearnKerasPipe.load_pipe(self._classifer_path)

    def predict(self, data):
        probas = self._model.predict_proba(data)
        return [dict(zip(self._classes, self.np_to_float(proba)))
                for proba in probas]

    @staticmethod
    def np_to_float(array):
        return [round(float(p), 2) for p in array]


class AnomalyDetector:

    _classes = {
        0: 'Abnormal',
        1: 'Normal',

    }

    _model_path = os.path.join(
        saved_model_path,
        'anomaly_detector',
        'anomaly_detector_100_dtw.joblib')

    def __init__(self):
        self._model = joblib.load(self._model_path)

    def predict(self, data):
        classes = self._model.predict(data)
        return [self._classes[class_] for class_ in classes]
