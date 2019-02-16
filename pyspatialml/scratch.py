Band = namedtuple(
    'Band',
    ['bidx', 'dtype', 'shape', 'nodata', 'file', 'read',
     'driver', 'crs', 'transform'])

band = Band(bidx=band.bidx,
            dtype=band.dtype,
            shape=band.shape,
            nodata=band.ds.nodata,
            file=band.ds.files,
            read=partial(band.ds.read, indexes=band.bidx),
            driver=band.ds.meta['driver'],
            crs=band.ds.crs,
            transform=band.ds.transform)

bands = [Band(*vals) for vals in zip(
    src.indexes,
    src.dtypes,
    src.shape,
    src.nodatavals,
    [src.files for i in range(src.count)],
    [src.read for i in range(src.count)]
)]