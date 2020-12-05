import numpy as np
import pytest
from numpy.testing import assert_array_equal

from toolbox.utils import (TimeSeriesResampler, TimeSeriesScalerMeanVariance,
                           dict_to_time_series, to_time_series_dataset,
                           to_dataset)


class TestToTimeSeriesDataset:

    def test_empty_dataset_raises_error(self):
        with pytest.raises(AssertionError):
            to_time_series_dataset([])

    def test_returns_right_dimension(self):
        datasets = [
            ([1],
             np.array([[[1.0]]])),
            ([1, 2],
             np.array([[[1.0], [2.0]]])),
            ([[1], [2]],
             np.array([[[1.0]], [[2.0]]])),
            ([[1, 2], [3, 4]],
             np.array([[[1.0], [2.0]], [[3.0], [4.0]]])),
            ([[[11, 12], [13, 14]], [[21, 22], [23, 24]]],
             np.array(
                [[[11.0, 12.0], [13.0, 14.0]],
                 [[21.0, 22.0], [23.0, 24.0]]]))
        ]
        for dataset, res in datasets:
            assert_array_equal(res, to_time_series_dataset(dataset))


class TestToDataSet:
    def test_empty_dataset_raises_error(self):
        with pytest.raises(AssertionError):
            to_dataset([])

    def test_returns_right_dimension(self):
        datasets = [
            ([1],
             [[[1]]]),
            #
            ([1, 2],
             [[[1], [2]]]),
            #
            ([[1, 2]],
             [[[1], [2]]]),
            #
            ([[1], [2]],
             [[[1]], [[2]]]),
            #
            ([[1, 2], [3, 4]],
             [[[1], [2]], [[3], [4]]]),
            #
            ([[[11, 12], [13, 14]], [[21, 22], [23, 24]]],
             [[[11, 12], [13, 14]],
              [[21, 22], [23, 24]]])
        ]
        for dataset, res in datasets:
            assert_array_equal(res, to_time_series_dataset(dataset))

    def test_different_features_raises_error(self):
        datasets = [
            [[1], [[2, 3]]],
            [[[11, 12], [13, 14]], [[21, 22], [23, 24, 25]]],
        ]
        for dataset in datasets:
            with pytest.raises(AssertionError):
                to_dataset(dataset)


def test_dict_to_timesereis():
    data = {
        'id': 'foo',
        'force': [1, 2, 3],
        'position': [1, 2, 3],
        'position_2': [1, 2, 3],
    }
    res = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
        ]
    )

    np.testing.assert_array_equal(
        dict_to_time_series(data),
        res)


class TestResampler:
    data = np.array(range(20)).reshape(2, 5, -1)

    def test_sampling(self):
        resampler = TimeSeriesResampler(4)
        np.testing.assert_allclose(
            resampler.transform([[1, 2]]),
            np.array([1.0, 1.333, 1.666, 2.0]).reshape(1, 4, 1),
            atol=1e-3
        )

    def test_shapes(self):
        resampler = TimeSeriesResampler(20)
        assert resampler.transform(self.data).shape == (2, 20, 2)

        resampler = TimeSeriesResampler(2)
        assert resampler.transform(self.data).shape == (2, 2, 2)


class TestScaler:

    data = np.array(
        [[[1, 2],
          [1, 2],
          [1, 2]],

         [[3, 4],
          [3, 4],
          [3,  4]]])

    def test_constant(self):
        scaler = TimeSeriesScalerMeanVariance(kind='constant')
        scaler.fit(self.data)
        assert scaler.mean_t.tolist() == [2., 3.]
        assert scaler.std_t.tolist() == [1., 1.]
        res = [[[-1., -1.],
                [-1., -1.],
                [-1., -1.]],
               [[1., 1.],
                [1., 1.],
                [1., 1.]]]
        assert scaler.transform(self.data).tolist() == res

    def test_timevarying(self):
        scaler = TimeSeriesScalerMeanVariance(kind='time-varying')
        scaler.fit(self.data)
        assert scaler.mean_t.tolist() == [[2., 3.], [2., 3.], [2., 3.]]
        assert scaler.std_t.tolist() == [[1., 1.], [1., 1.], [1., 1.]]
