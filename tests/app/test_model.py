import numpy as np
from app.model import Classifier


class TestClassifier:

    def test_runs(self):
        classifier = Classifier()
        data = {
            'force': np.random.rand(500).tolist(),
            'position': np.random.rand(500).tolist(),
        }
        assert classifier.predict(data)
