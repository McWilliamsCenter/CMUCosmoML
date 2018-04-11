from astropy.table import Table, vstack, join
from sklearn.preprocessing import RobustScaler
import numpy as np

def preprocess_data(quasar_obs_table, nonquasar_obs_table,
                    output_filename='peters_2015_data.fits.gz',
                    max_length=80, min_length=10):
    """
    This function preprocesses the input astropy tables into a
    a time series format with fixed maximum length.
    """
    # Select the features to include in the time-series
    features = ['mjd_std', 'u_std', 'g_std', 'r_std', 'i_std', 'z_std',
                'psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 
                'psfMagErr_i', 'psfMagErr_z', 'airmass_std']
    
    # Open and merge the two observation tables
    quasar = Table.read(quasar_obs_table)
    quasar['label'] = 1

    non_quasar = Table.read(nonquasar_obs_table)
    non_quasar['label'] = 0

    dataset = vstack([quasar, non_quasar])
    
    # Remove unobserved/partially observed row
    inds = np.where((dataset['mjd_u'] == 0) + (dataset['mjd_g'] == 0) + 
                    (dataset['mjd_r'] == 0) + (dataset['mjd_i'] == 0) +
                    (dataset['mjd_z'] == 0))
    dataset.remove_rows(inds[0])

    # Computes extra standardized columns
    dataset['mjd_std'] = (dataset['mjd_u'] + dataset['mjd_g'] + 
                          dataset['mjd_r'] + dataset['mjd_i'] + 
                          dataset['mjd_z']) / 5 / 3000.
    dataset['airmass_std'] = RobustScaler().fit_transform(dataset['airmass_r'].reshape(-1, 1)).squeeze()
    for b in ['u', 'g', 'r', 'i', 'z']:
        dataset['%s_std'%b] = RobustScaler().fit_transform(
            (dataset[b] - dataset['coadd_%s' % b]).reshape(-1, 1)).squeeze()

    # Sort the table by observation and time
    dataset.sort(['objid', 'mjd_std'])

    # count the number of observations for each entries
    unique_ids, idices, counts= np.unique(dataset['objid'], return_inverse=True, return_counts=True)
    dataset['obs_len'] = counts[idices]

    # Remove observations with fewer than minimum number of obs
    dataset = dataset[dataset['obs_len'] > min_length]

    # Recount after the last operation
    unique_ids, idices, counts= np.unique(dataset['objid'], return_inverse=True, return_counts=True)
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
    unique_ids, idices, counts, inde= np.unique(dataset['objid'], 
                                                return_inverse=True, 
                                                return_counts=True,
                                                return_index=True)
    dataset_table = dataset[idices]
    
    # Join and save the table
    output_table = Table(join(dataset_table, time_series_table), masked=False)
    output_table.write(output_filename)
    