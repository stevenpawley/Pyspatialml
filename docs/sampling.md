# Random Sampling

## Random Uniform Sampling

For many spatial models, it is common to take a random sample of the
predictors to represent a single class (i.e. an environmental background or
pseudo-absences in a binary classification model). The sample function is
supplied in the sampling module for this purpose:

```
# extract training data using a random sample
df_rand = stack.sample(size=1000, random_state=1)
df_rand.plot()
```

## Stratified Random Sampling

The sample function also enables stratified random sampling based on passing a
categorical raster dataset to the strata argument. The categorical raster
should spatially overlap with the dataset to be sampled, but it does not need
to be of the same grid resolution. This raster should be passed as a opened
rasterio dataset:

```
with rasterio.open(nc.strata) as strata:

    df_strata = stack.sample(size=5, strata=strata, random_state=1)
    df_strata = df_strata.dropna()

    fig, ax = plt.subplots()

    ax.imshow(
        data=strata.read(1, masked=True),
        extent=rasterio.plot.plotting_extent(strata),
        cmap='tab10')

    df_strata.plot(ax=ax, markersize=20, color='white')
    plt.show()
```