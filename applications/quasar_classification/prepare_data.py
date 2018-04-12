# This script creates a quasar light curve dataset from stripe 82
import wget
from sciserver import casjobs, authentication
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
import numpy as np
import pandas as pd
import os.path
import argparse

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
BATCH_SIZE=20000

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
        s82_coadd = casjobs.executeQuery(sql_query.read(), context='stripe82')

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
        job_id = casjobs.submitJob("drop table ref_cat", context='MyDB')
        casjobs.waitForJob(job_id)
        job_id = casjobs.submitJob("drop table tmp_table", context='MyDB')
        casjobs.waitForJob(job_id)
    except e:
        print("error", e)

    for i in n_batch:
        print("Downloading batch ",i)
        # Upload the piece of reference catalog
        casjobs.uploadPandasDataFrameToTable(ref_cat[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 'ref_cat', context='MyDB')
        print("Index uploaded")
        # Submit job to collect light curves
        with open(S82_LC_QUERY, 'r') as sql_query:
            job_id = casjobs.submitJob(sql_query.read(), context='stripe82')
        print("SQL query submitted")
        # Wait for job to finish
        casjobs.waitForJob(job_id)
        print("Retrieving light curves...")
        # Retrieve ligth curve dataset
        res = casjobs.executeQuery("select * from tmp_table", context='MyDB')
        print("Done :-)")
        if s82_lightcurves is None:
            s82_lightcurves = res
        else:
            s82_lightcurves.append(res, ignore_index=True)

        # Drop both the temp table and reference catalog to start fresh
        job_id = casjobs.submitJob("drop table ref_cat", context='MyDB')
        casjobs.waitForJob(job_id)
        job_id = casjobs.submitJob("drop table tmp_table", context='MyDB')
        casjobs.waitForJob(job_id)

    # Writes down the main coadded table as well as the light curve data
    s82_coadd.to_hdf('s82_pointsource_catalog.hdf', 'coadd', mode='w')
    s82_coadd.to_hdf('s82_pointsource_catalog.hdf', 'lightcurve', mode='a')
