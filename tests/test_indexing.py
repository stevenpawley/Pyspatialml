from unittest import TestCase
from pyspatialml import Raster, RasterLayer
from pyspatialml.datasets import nc


class TestIndexing(TestCase):

    predictors = [nc.band1, nc.band2, nc.band3, nc.band4, nc.band5, nc.band7]

    def test_naming(self):

        # check unique naming when stacking multiband raster
        stack = Raster(self.predictors + [nc.multiband])
        self.assertEqual(stack.count, 11)
        expected_names = [
            'lsat7_2000_10',
            'lsat7_2000_20',
            'lsat7_2000_30',
            'lsat7_2000_40',
            'lsat7_2000_50',
            'lsat7_2000_70',
            'landsat_multiband_1',
            'landsat_multiband_2',
            'landsat_multiband_3',
            'landsat_multiband_4',
            'landsat_multiband_5'
        ]
        self.assertListEqual(stack.names, expected_names)

    def test_subseting(self):

        stack = Raster(self.predictors + [nc.multiband])

        # RasterLayer indexing which returns a RasterLayer
        self.assertIsInstance(stack.iloc[0], RasterLayer)
        self.assertIsInstance(stack.loc['lsat7_2000_10'], RasterLayer)
        self.assertIsInstance(stack.lsat7_2000_10, RasterLayer)
        self.assertListEqual(
            stack.iloc[0:3],
            stack.loc[['lsat7_2000_10', 'lsat7_2000_20', 'lsat7_2000_30']])

        # RasterStack subsetting
        subset_raster = stack[['lsat7_2000_10', 'lsat7_2000_70']]
        self.assertListEqual(subset_raster.names, ['lsat7_2000_10', 'lsat7_2000_70'])
        self.assertEqual(subset_raster.lsat7_2000_10.read(masked=True).mean(), 80.56715262406088)
        self.assertEqual(subset_raster.lsat7_2000_70.read(masked=True).mean(), 59.17773813401238)

        # subsetting after name change
        stack.rename({'lsat7_2000_10': 'testme'})
        expected_names = [
            'lsat7_2000_20',
            'lsat7_2000_30',
            'lsat7_2000_40',
            'lsat7_2000_50',
            'lsat7_2000_70',
            'landsat_multiband_1',
            'landsat_multiband_2',
            'landsat_multiband_3',
            'landsat_multiband_4',
            'landsat_multiband_5',
            'testme']
        self.assertListEqual(stack.names, expected_names)
        self.assertListEqual(stack.iloc[-1].names, ['testme'])
        self.assertListEqual(stack[['testme', 'lsat7_2000_20']].names, ['testme', 'lsat7_2000_20'])

        # check that RasterLayer internal name is carried over to new Raster
        self.assertListEqual(Raster(stack.iloc[-1]).names, ['testme'])

    def test_indexing(self):

        stack = Raster(self.predictors + [nc.multiband])

        # replace band 1 with band 7
        band7_mean = stack.loc['lsat7_2000_70'].read(masked=True).mean()

        stack.iloc[0] = Raster(nc.band7).iloc[0]
        self.assertEqual(stack.iloc[0].read(masked=True).mean(), band7_mean)
        self.assertEqual(stack.loc['lsat7_2000_10'].read(masked=True).mean(), band7_mean)
        self.assertEqual(stack['lsat7_2000_10'].read(masked=True).mean(), band7_mean)
        self.assertEqual(stack.lsat7_2000_10.read(masked=True).mean(), band7_mean)

        # append another Raster containing a single layer
        stack.append(Raster(nc.band7))
        self.assertEqual(stack.names[5], 'lsat7_2000_70_1')
        self.assertEqual(stack.names[-1], 'lsat7_2000_70_2')
        self.assertEqual(stack.lsat7_2000_70_1.read(masked=True).mean(),
                         stack.lsat7_2000_70_2.read(masked=True).mean(),
                         band7_mean)

        # append a multiband raster
        stack.append(Raster(nc.multiband))
        self.assertEqual(stack.names[6], 'landsat_multiband_1_1')
        self.assertEqual(stack.names[12], 'landsat_multiband_1_2')

        # rename layers
        band3_mean = stack.lsat7_2000_30.read(masked=True).mean()
        stack.rename({'lsat7_2000_30': 'new_name'})
        self.assertEqual(stack['new_name'].read(masked=True).mean(), band3_mean)
        self.assertEqual(stack.new_name.read(masked=True).mean(), band3_mean)
        self.assertEqual(stack.loc['new_name'].read(masked=True).mean(), band3_mean)
        self.assertEqual(stack.iloc[-1].read(masked=True).mean(), band3_mean)

        # Drop a layer
        stack.names
        stack.drop(labels='lsat7_2000_50')
        self.assertNotIn('lsat7_2000_50', stack.names)
