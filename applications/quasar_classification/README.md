# Classification of photometric timeseries using Recurrent Neural Networks

## Prerequisites

To create the dataset, you need at least the following python packages:
```sh
  $ pip install wget sciserver pandas scikit-learn astropy
```

## Create the dataset

To download the data, you first need to create an account on [SciServer](http://www.sciserver.org/). Then you can run the following script:
```
 $ python prepare_data.py --user [your username] --pwd [your password]
```

This will do the following:
  - Download the SDSS DR14 Quasar catalog
  - Create and download a table of coadded point sources in Stripe82
  - Cross match that table with individual exposures to create light curves and
  with the quasar catalog to create labels
  - Creates a table with co-added info and light curves of fixed maximum length

The final file produced by this script is `ligthcurve_data.fits.gz` which can be
read using the following:
```python
>>> from astropy.table import Table
>>> data = Table.read('ligthcurve_data.fits.gz')
```
Here are a few relevant columns:
 - `time_series`: array storing the light curves, first dimension is a normalised
 mjd
 - `obs_len`: Length of the time series
 - `coadd_label`: 1 for confirmed quasars, 0 otherwise
