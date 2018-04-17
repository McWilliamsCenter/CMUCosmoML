# This script creates a quasar light curve dataset from stripe 82
import wget
from sciserver import casjobs, authentication
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, vstack, join
import numpy as np
import pandas as pd
import os.path
import argparse
from sklearn.preprocessing import RobustScaler

# Default path to the downloaded data
DATA_PATH="."

# This is our reference quasar catalog, described in arXiv:1712.05029
DR14Q_URL="https://data.sdss.org/sas/dr14/eboss/qso/DR14Q/DR14Q_v4_4.fits"

# Query for coadded catalog
S82_COADD_QUERY="stripe82_coadd.sql"
# Query for light curve catalog
S82_LC_QUERY="stripe82_lc.sql"

# Total length of the dataset
DATASET_LENGTH=100000

# Batch size for downloading light curves
BATCH_SIZE=10000

def cross_match(coadd_catalog, match_dist=1*u.arcsec):
    """
    Cross match the coadded point source catalog with the reference quasar
    catalog.
    """
    # Loads the qso table
    qso = Table.read('DR14Q_v4_4.fits')
    c_data = SkyCoord(coadd_catalog['ra'], coadd_catalog['dec'],
                      unit=(u.degree, u.degree), frame='icrs')

    c_qso = SkyCoord(qso['RA'], qso['DEC'],
                     unit=(u.degree, u.degree), frame='icrs')

    idx, d2, d3 = c_data.match_to_catalog_sky(c_qso)

    # Apply tolerance cuts
    tol_idx = np.array(range(len(c_data)))
    tol_nidx = tol_idx[d2 > match_dist]
    tol_idx = tol_idx[d2 <= match_dist]

    # Adds quasar label to coadd_catalog
    coadd_catalog['label'] = 0
    coadd_catalog['label'].iloc[tol_idx] = 1
    return coadd_catalog

def preprocess_data(input_filename='s82_pointsource_catalog.hdf',
                    output_filename='ligthcurve_data.fits.gz',
                    max_length=90, min_length=10, distance_cut=0.3 # arcsec
                    ):
    """
    This function preprocesses the raw sdss tables into a
    a time series format with fixed maximum length.
    """
    # Select the features to include in the time-series
    features = ['mjd_std',
                'psfMag_u_std', 'psfMag_g_std', 'psfMag_r_std', 'psfMag_i_std', 'psfMag_z_std',
                'psfmagerr_u', 'psfmagerr_g', 'psfmagerr_r', 'psfmagerr_i', 'psfmagerr_z', 'distance']

    dataset = Table.from_pandas(pd.read_hdf('s82_pointsource_catalog.hdf','/lightcurve'))
    coadd_data = Table.from_pandas(pd.read_hdf('s82_pointsource_catalog.hdf','/coadd'))

    for c in coadd_data.colnames:
        coadd_data[c].name = 'coadd_'+c

    dataset = join(dataset, coadd_data)

    # Remove unobserved/partially observed row
    inds = np.where((dataset['mjd_u'] == 0) + (dataset['mjd_g'] == 0) +
                    (dataset['mjd_r'] == 0) + (dataset['mjd_i'] == 0) +
                    (dataset['mjd_z'] == 0))
    dataset.remove_rows(inds[0])

    # Remove rows deviating more than 0.3 arcsec from coadded position
    inds = np.where((dataset['distance']*60 > distance_cut))
    dataset.remove_rows(inds[0])

    # Corrects all magnitudes for extinction
    for b in ['u', 'g', 'r', 'i', 'z']:
        dataset['psfMag_%s'%b] = dataset['psfMag_%s'%b] - dataset['extinction_%s'% b]
        dataset['coadd_psfMag_%s'%b] = dataset['coadd_psfMag_%s'%b] - dataset['coadd_extinction_%s'% b]


    # Computes extra standardized columns
    dataset['mjd'] = ((dataset['mjd_u'] + dataset['mjd_g'] +
                          dataset['mjd_r'] + dataset['mjd_i'] +
                          dataset['mjd_z']) / 5).astype('float64')

    dataset['mjd_std'] = dataset['mjd'] / 3000.

    for b in ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']:
        dataset['%s_std'%b] = RobustScaler().fit_transform(
            (dataset[b] - dataset['coadd_%s' % b]).reshape(-1, 1)).squeeze()

    dataset['distance'] = RobustScaler().fit_transform(dataset['distance'].reshape(-1, 1)).squeeze()

    # Sort the table by observation and time
    dataset.sort(['coadd_objid', 'mjd_std'])

    # count the number of observations for each entries
    unique_ids, idices, counts= np.unique(dataset['coadd_objid'], return_inverse=True, return_counts=True)
    dataset['obs_len'] = counts[idices]

    # Remove observations with fewer than minimum number of obs
    dataset = dataset[dataset['obs_len'] > min_length]

    # Recount after the last operation
    unique_ids, idices, counts= np.unique(dataset['coadd_objid'], return_inverse=True, return_counts=True)
    dataset['obs_len'] = counts[idices]
    dataset['obs_id'] = np.arange(len(counts))[idices]

    # Create array to store the time series
    ts = np.zeros((len(counts), max_length, len(features))).astype('float32')

    # Restrict to the features we want to extract
    d = dataset[features]
    ind=0
    for i in range(len(counts)):
        if i %100 == 0:
            print("processed %d entries"%i)
            assert dataset[ind]['obs_id'] == i
        t = dataset[ind:ind+dataset[ind]['obs_len']][features]

        ind += dataset[ind]['obs_len']

        t = t[0:min(len(t),max_length)]
        ts[i,:len(t),:] = np.array(t).view(np.float64).reshape((-1,len(features))).astype(np.float32)
        ts[i,:,0] -= ts[i,0,0] # Subtract the first mjd

    # Create a table to save the time series
    time_series_table = Table({'time_series':ts, 'obs_id':range(len(ts))})

    # Create a small data table with just the first observation of each object
    unique_ids, idices, counts, inde= np.unique(dataset['coadd_objid'],
                                                return_inverse=True,
                                                return_counts=True,
                                                return_index=True)
    dataset_table = dataset[idices]

    # Join and save the table
    output_table = Table(join(dataset_table, time_series_table), masked=False)
    output_table.write(output_filename, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a quasar light curve dataset.')
    parser.add_argument('--user', help='sciserver login username', required=True)
    parser.add_argument('--pwd', help='sciserver login password', required=True)

    args = parser.parse_args()

    # 1. Download reference quasar catalog
    if not os.path.exists('DR14Q_v4_4.fits'):
        wget.download(DR14Q_URL)

    # 2. Create and download catalog of coadded point sources
    token = authentication.login(args.user, args.pwd)     # Login to sciserver
    with open(S82_COADD_QUERY, 'r') as sql_query:
        job_id = casjobs.submitJob(sql_query.read(), context='stripe82')
    casjobs.waitForJob(job_id)
    s82_coadd = casjobs.executeQuery("select * from tmp_table", context='MyDB')
    casjobs.executeQuery("drop table tmp_table", context='MyDB')

    # 3. Cross-match with reference quasar catalog to get all known quasars
    s82_coadd = cross_match(s82_coadd)

    # 4. Create random subsample
    s82_coadd_nq = s82_coadd[s82_coadd['label'] == 0]
    s82_coadd_q = s82_coadd[s82_coadd['label'] == 1]
    # Subsample the non quasars to keep the dataset manageable
    s82_coadd_nq = s82_coadd_nq.sample( DATASET_LENGTH - len(s82_coadd_q),
                                        random_state=1234)
    # Merge back both datasets
    s82_coadd =  s82_coadd_q.append(s82_coadd_nq, ignore_index=True)


    # 5. Extract light curves for all objects left in our coadd catalog
    # Upload table with just the objid, ra, dec we care about
    ref_cat = s82_coadd[['objid', 'ra', 'dec']]

    # Loop through this reference catalog to download light curves as it won't
    # fit in one query
    n_batch = len(ref_cat) // BATCH_SIZE
    s82_lightcurves = None

    # Try to drop the temporary tables we will be creating
    print("Dropping temporary tables if they exist")
    try:
        casjobs.executeQuery("drop table my_cat", context='MyDB')
        casjobs.executeQuery("drop table my_table", context='MyDB')
    except:
        print("error")

    for i in range(n_batch):
        print("Downloading batch ", i)

        # Upload the piece of reference catalog
        casjobs.uploadPandasDataFrameToTable(ref_cat[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 'my_cat', context='MyDB')
        print("Index uploaded")

        # Submit job to collect light curves
        with open(S82_LC_QUERY, 'r') as sql_query:
            job_id = casjobs.submitJob(sql_query.read(), context='stripe82')
        print("SQL query submitted")
        # Wait for job to finish
        casjobs.waitForJob(job_id)
        print("Retrieving light curves...")
        # Retrieve ligth curve dataset
        res = casjobs.executeQuery("select * from my_table", context='MyDB')
        print("Done :-)")

        if s82_lightcurves is None:
            s82_lightcurves = res
        else:
            print("Appending data")
            s82_lightcurves = s82_lightcurves.append(res, ignore_index=True)

        # Drop both the temp table and reference catalog to start fresh
        casjobs.executeQuery("drop table my_cat", context='MyDB')
        casjobs.executeQuery("drop table my_table", context='MyDB')

    # Writes down the main coadded table as well as the light curve data
    s82_coadd.to_hdf('s82_pointsource_catalog.hdf', 'coadd', mode='w')
    s82_lightcurves.to_hdf('s82_pointsource_catalog.hdf', 'lightcurve', mode='a')

    # Creates lightcurves in a simple format
    preprocess_data()
