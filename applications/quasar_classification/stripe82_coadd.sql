-- This sql query creates a database of stripe82 coadded point sources
SELECT
  objid, ra, dec, run, field,
  psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,
  psfmagerr_u, psfmagerr_g, psfmagerr_r, psfmagerr_i, psfmagerr_z,
  extinction_u, extinction_g, extinction_r, extinction_i, extinction_z,
  flags
INTO tmp_table
FROM
  PhotoObjAll
WHERE ((flags & 0x10000000) != 0) -- detected in BINNED1
  AND ((flags & 0x8100000c00a4) = 0) -- not EDGE, NOPROFILE, PEAKCENTER,
                                     -- NOTCHECKED, PSF_FLUX_INTERP,
                                     -- SATURATED, or BAD_COUNTS_ERROR
  AND (((flags & 0x400000000000) = 0) or
    (psfmagerr_r <= 0.2 and psfmagerr_i<= 0.2 and psfmagerr_g<=0.2 and psfmagerr_u<=0.2 and psfmagerr_z<=0.2))
                -- not DEBLEND_NOPEAK or small PSF error
                -- (substitute psfmagerr in other band as appropriate)
  AND (((flags & 0x100000000000) = 0) or (flags & 0x1000) = 0)
     -- not INTERP_CENTER or not COSMIC_RAY omit this AND clause if you
     -- want to accept objects with interpolation problems for PSF mags.
  AND (run = 106 or run = 206) -- Select the coadded runs
  AND type = 6  -- Stars, or at least point sources
  AND mode = 1  -- Primary detections
  AND ((RA BETWEEN 0 AND 55 ) OR (RA BETWEEN 320 AND 360))  -- Basic cuts from Peter et al. 2015
  AND psfMag_g < 23.5 AND psfMag_i < 22. AND psfMag_g - psfMag_i < 6.0
