from unittest import TestCase
from pyspatialml.transformers import AspectTransformer, KNNTransformer

import numpy as np

class TestTransformers(TestCase):
    def test_aspect_transformer(self):
        trans = AspectTransformer()
        dirs = np.arange(0, 360, 1, dtype=np.float32)

        mag = trans.fit_transform(dirs)
        inverse = trans.inverse_transform(mag)
        inverse = inverse.round(0)

        self.assertListEqual(dirs.tolist(), inverse.tolist())

    def test_knn_transformer_excludes_zero_distance_self(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([100.0, 2.0, 4.0, 6.0])
        transformer = KNNTransformer(
            n_neighbors=1, weights="uniform", normalize=False
        )

        transformed = transformer.fit_transform(X, y)

        self.assertEqual(transformed[0, -1], 2.0)

    def test_knn_transformer_uniform_weights(self):
        X = np.array([[0.0], [2.0], [4.0], [6.0]])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        transformer = KNNTransformer(
            n_neighbors=2, weights="uniform", normalize=False
        )

        transformed = transformer.fit_transform(X, y)

        self.assertEqual(transformed.shape, (4, 2))
        self.assertEqual(transformed[0, -1], 3.0)

    def test_knn_transformer_multioutput(self):
        X = np.array([[0.0], [2.0], [4.0], [6.0]])
        y = np.column_stack((X[:, 0], 10.0 + X[:, 0]))
        transformer = KNNTransformer(
            n_neighbors=2, weights="uniform", normalize=False
        )

        transformed = transformer.fit_transform(X, y)

        self.assertEqual(transformed.shape, (4, 3))
        np.testing.assert_allclose(transformed[0, -2:], [3.0, 13.0])
