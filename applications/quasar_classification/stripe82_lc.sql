SELECT
  p1.objid as coadd_objid, p2.objid, p2.ra, p2.dec, p2.run,
  mjd_u, mjd_g, mjd_r, mjd_i, mjd_z,
  p2.psfMag_u, p2.psfMag_g, p2.psfMag_r, p2.psfMag_i, p2.psfMag_z,
  p2.psfmagerr_u, p2.psfmagerr_g, p2.psfmagerr_r, p2.psfmagerr_i, p2.psfmagerr_z,
  p2.extinction_u, p2.extinction_g, p2.extinction_r, p2.extinction_i, p2.extinction_z,
  m.distance
INTO my_table
FROM
  mydb.my_cat p1
  CROSS APPLY dbo.fGetNearbyObjEq(p1.ra, p1.dec, 0.016) AS m
  join PhotoObjAll p2 on p2.objid = m.objid
  join Field f on p2.fieldid = f.fieldid
where p2.type = 6 AND p2.mode = 1
  AND (p2.run != 106) AND (p2.run != 206)
  AND ((p2.flags & 0x10000000) != 0)
  AND ((p2.flags & 0x8100000c00a4) = 0)
  AND (((p2.flags & 0x400000000000) = 0) or
    (p2.psfmagerr_r <= 0.2 and p2.psfmagerr_i<= 0.2 and p2.psfmagerr_g<=0.2 and p2.psfmagerr_u<=0.2 and p2.psfmagerr_z<=0.2))
  AND (((p2.flags & 0x100000000000) = 0) or (p2.flags & 0x1000) = 0)
  order by coadd_objid
