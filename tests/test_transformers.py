from unittest import TestCase
from pyspatialml.transformers import AspectTransformer

import numpy as np

class TestTransformers(TestCase):
    def test_aspect_transformer(self):
        trans = AspectTransformer()
        dirs = np.arange(0, 360, 1, dtype=np.float32)

        mag = trans.fit_transform(dirs)
        inverse = trans.inverse_transform(mag)
        inverse = inverse.round(0)

        self.assertListEqual(dirs.tolist(), inverse.tolist())
