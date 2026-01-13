'''
 The class holding the information on a single swatch of the MODIS satellite instrument

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

Functions available:
- __init__:   Sets abasic file names 
- get_pixel_size:  basic MODIS geometry
- draw_pixel_size: draw the main MODIS geometry, in fact, reference to granule_basic
- MODIS_1km_geo_from_5km: Takes the so-called reduced-resolution fields of longitude and latitude
                          and makes full-resolutionn ones for 1km
- unpack_MxD03_byte_1: Bit fields within each byte are numbered from the left: 7, 6, 5, 4, 3, 2, 1, 0.
- unpack_MxD14_QA:     unpacks the quality field. NotImplemented
- pick_granule_data_IS4FIRES_v3_0: gets the needed data from MxD14 and MxD35
######- detection_limit;     returns the detection limit map as a function of the day/night flag and pixel size
- pick_granule_data_IS4FIRES_v2_0: gets data needed for old IS4FIRES version from MOD14
- write_granule_IS4FIRES_v2_0:  write the granule down as IS4FREIS 2.0 need
- draw_granule:        Draws area covered by this swath, with a bit free space around:

@author: sofievm
'''
# A couple of constants

import numpy as np, datetime as dt
import glob, os, sys
from support import netcdftime
import matplotlib as mpl
import netCDF4 as nc4
from mpl_toolkits import basemap
import granule__basic as gb
from toolbox import supplementary as spp
import fire_records
try:
    from pyhdf.SD import SD, SDC
except:
    print('pyhdf failed to load')  

ifDebug = False


########################################################################
#
# MODIS PIXEL SIZE LUT
# Fit to the  "approximate but quite accurate formula" implemented in the granule below. This is just a fast access
#
#Nmod = 1354
#refdSm = [0,3,6,10,14,18,22,26,31,36,41,46,51,57,63,69,76,83,90,98,106,115,124,134,144,155,167,179,192,206,221,237,254,272,292,313,336,360,386,414,444,476,510,546,584,624,664]
#valdSm = [4.8239,4.7126,4.6062,4.4713,4.3439,4.2234,4.1091,4.0007,3.8728,3.7525,3.6392,3.5324,3.4315,3.3176,3.2108,3.1105,3.0010,2.8988,2.8032,2.7013,2.6065,2.5076,2.4159,2.3216,2.2346,2.1464,2.0580,1.9771,1.8968,1.8179,1.7412,1.6671,1.5961,1.5286,1.4615,1.3991,1.3387,1.2836,1.2317,1.1837,1.1402,1.1017,1.0686,1.0414,1.0205,1.0065,1.0004]
##refptsterp [ 3  3  4  4  4  4  4  5  5  5  5  5  6  6  6  7  7  7  8  8  9  9 10 10
## 11 12 12 13 14 15 16 17 18 20 21 23 24 26 28 30 32 34 36 38 40 40]
#refdTm = [0,12,25,40,56,73,92,113,136,161,188,218,251,287,326,369,416,466,520,576,634]
#valdTm = [2.0046,1.9336,1.8643,1.7925,1.7240,1.6588,1.5939,1.5301,1.4684,1.4094,1.3536,1.2996,1.2484,1.2006,1.1569,1.1169,1.0814,1.0518,1.0281,1.0114,1.0020]
##refptsterp [12 13 15 16 17 19 21 23 25 27 30 33 36 39 43 47 50 54 56 58]
#refdSmFull = np.concatenate( (np.array(refdSm), Nmod - 1 -  np.flip(np.array(refdSm)),) )
#valdSmFull = np.concatenate( (np.array(valdSm), np.flip(np.array(valdSm)),) )
#refdTmFull = np.concatenate( (np.array(refdTm), Nmod - 1 -  np.flip(np.array(refdTm)), ))
#valdTmFull = np.concatenate( (np.array(valdTm), np.flip(np.array(valdTm)),))
#
#def modis_pixel_size_LUT(ipix = None):
#    if ipix is None: ipix = np.arange(Nmod)
#    dS = np.interp(ipix, refdSmFull, valdSmFull)
#    dT = np.interp(ipix, refdTmFull, valdTmFull)
#    return dS, dT

#=============================================================

def productType(chFNm):
    dicKnownProducts = {'MOD03':('MODIS','auxiliary'), 'MYD03':('MODIS','auxiliary'),
                        'MOD35':('MODIS','auxiliary'), 'MYD35':('MODIS','auxiliary'),
                        'MOD14':('MODIS','fire'),      'MYD14':('MODIS','fire')}
    for satProd in dicKnownProducts.keys():
        if satProd in chFNm: return dicKnownProducts[satProd]
    return 'unknown'



#################################################################################
#
# The class holding the information on a single granule of the satellite, 
# ~ 2030 x 1354 pixels and 5 min of the orbit for MODIS
#
#################################################################################

class granule_MODIS(gb.granule_basic):

    #=============================================================
    def __init__(self, now_UTC=dt.datetime.utcnow(), 
                 chFRPfilesTempl='', chAuxilFilesTempl='', log=None):
        self.type = 'MODIS'
        gb.granule_basic.__init__(self, self.type, now_UTC, chFRPfilesTempl, chAuxilFilesTempl, log,
                                  (log is None) or (not ifDebug))
        self.get_pixel_size()        # get the distribution of MODIS swath


    #=============================================================

    def get_pixel_size(self):
        #
        # An approximate but quite accurate formula for dS and dT pixel size along the scan 
        # and along the track directions, respectively
        # Presented in Ichoku & Kaufman, IEEE Transactions 2005. 
        # dS error: min_LUT-fla, max_LUT-fla, max_rel: -0.0152738 km -3.88914e-07 km 0.00318493
        # dT error: min_LUT-fla, max_LUT-fla, max_rel: 1.64066e-07 km 0.000403797 km 0.000201632
        # theta error: min_LUT-fla, max_LUT-fla, max_rel: -5.674e-12 rad 5.651e-12 rad 1.04e-08
        #
        self.N = 1354      # number of pixels in a scan line
        self.h = 705
        self.scanSz = 10
        R_Earth = gb.R_Earth
#        s = 0.0014184397  # ratio of nadir pixel size to h if calculated
        s = 0.001418   # ratio of nadir pixel size to h used in the LUT
        r = R_Earth + self.h
        self.theta = np.array( list( ((s * (0.5*(-self.N+1) + i)) for i in range(self.N)) ) )
        sinTmp = np.sin(self.theta) 
        cosTmp = np.cos(self.theta)
        aTmp = np.sqrt((R_Earth/ r)**2 - np.square(sinTmp))
        # distance from swatch centre point
        c = r * cosTmp - np.sqrt( r*r* np.square(cosTmp) + R_Earth*R_Earth - r*r)
        self.dist_nadir = R_Earth * np.arcsin(c * sinTmp / R_Earth)
        # pixel sizes
        self.dS = R_Earth * s * ((cosTmp / aTmp) - 1.0)   # size along scan
        self.dT = r * s * (cosTmp - aTmp)            # size along track
        self.area = self.dS * self.dT
        # Overlap along the scan direction
        self.dist_pixels_S = self.dist_nadir[1:] - self.dist_nadir[:-1]   # minus one element
        self.overlap_S = ((self.dS[1:] + self.dS[:-1]) * 0.5 / self.dist_pixels_S - 1)  # fraction
        # overlap along track is based on 2063 km length of a single swatch (approx. value)
        self.overlap_T = (self.dT - (2063.0 / 2030)) / self.dT   # fraction
#        self.area_corr = self.area / (1. - self.overlap_T)   # did not find the use case
        #
        # make a map of overlapping pixels for a single scan
        # cell centers of the edge of the previous scan projected to the current one (-0.5 in the middle of scan)
        edge_prev_scan_along_track = -1 + self.overlap_T * self.scanSz/2  # from -1
        edge_next_scan_along_track = self.scanSz - self.overlap_T * self.scanSz/2  # up to scanSz
        # overlap for pixels
        self.overlap_pattern = np.ones(shape=(self.N, self.scanSz), dtype=np.float32)  # one scan
        for iTrac in range(self.scanSz):
            self.overlap_pattern[edge_prev_scan_along_track >= iTrac, iTrac] = 2  # overlap with prev scan
            self.overlap_pattern[edge_next_scan_along_track <= iTrac, iTrac] = 2  # overlap with next scan
            self.overlap_pattern[np.logical_and(edge_next_scan_along_track <= iTrac,
                                                edge_prev_scan_along_track >= iTrac), 
                                                iTrac] = 3  # two overlaps

    
    #=============================================================
    def abbrev_1(self, chFNm):
        if os.path.basename(chFNm).startswith('MOD'):
            return 'O'.encode('utf-8')
        elif os.path.basename(chFNm).startswith('MYD'):
            return 'Y'.encode('utf-8')
        else: raise ValueError('Cannot determine the satellite from name:', chFNm)


    #=============================================================

    def MODIS_1km_geo_from_5km(self, lon_5km, lat_5km):
        #
        # Takes the so-called reduced-resolution fields of longitude and latitude
        # and makes full-resolutionn ones for 1km
        #
        # Fill in the known 5km locations
#        i5kmIn1km = np.array(range(lon_5km.shape[0])) * 5 + 2
#        j5kmIn1km = np.array(range(lon_5km.shape[1])) * 5 + 2
#        xx5kmIn1km, yy5kmIn1km = np.meshgrid(j5kmIn1km, i5kmIn1km)     # transpose
#        xx1km, yy1km = np.meshgrid(np.array(range(lon_5km.shape[1]*5+4)),
#                                   np.array(range(lon_5km.shape[0]*5)))
        # check that the field is smooth. Longitude can jump over 180 meridian
        #
        # Take care of -180 longitude: cannot loow the break
        # interpolate between them
        ifJump = False
        if np.nanmax(np.abs(lon_5km[1:,:] - lon_5km[:-1,:])) > 10.:
            idxJumpLon = (np.abs(lon_5km[1:,:-1] - lon_5km[:-1,:-1]) +
                          np.abs(lon_5km[:-1,1:] - lon_5km[:-1,:-1])) > 200.
#            print('Suspected longitude break, ', np.max(np.abs(lon_5km[1:,:] - lon_5km[:-1,:])))
#            print(lon_5km.shape, np.sum(idxJumpLon))
#            print(self.now_UTC)
#            self.log.log('\nIndication af jump long x, longitude 5km field')
#            for i in range(lon_5km.shape[0]-1):
#                self.log.log( (' '.join(str(idxJumpLon[i,jTmp]) for jTmp in range(idxJumpLon.shape[1]))))
#            self.log.log('\nFull longitude 5km field')
#            for i in range(lon_5km.shape[0]):
#                self.log.log( (' '.join('%6.2f' % lon_5km[i,jTmp] for jTmp in range(lon_5km.shape[1]))))
#            print('\nIndices of the jump:')
#            print(idxJumpLon)
#            print('\nLongitude of the indices')
#            print(lon_5km[1:,:-1][idxJumpLon])
#            print('\n...and their neighbours')
#            print(lon_5km[:-1,:-1][idxJumpLon])
#            print('\n\n\n')

            ifJump = True
        #
        # If input fields contain missing data we should not our output for nans
        ifInputClean = np.all(np.isfinite(lon_5km)) and np.all(np.isfinite(lat_5km))

        # Alternative way: regular grid filling is faster than generic interpolation
        geo_1km = (np.zeros(shape=(lon_5km.shape[0]*5,lon_5km.shape[1]*5+4)) * np.nan,   # lon 1km
                   np.zeros(shape=(lon_5km.shape[0]*5,lon_5km.shape[1]*5+4)) * np.nan)   # lat 1km

        # both longitude and latitude are interpolated the same way
        # Initialise...
        geo_1km[0][2::5, 2:-4:5] = lon_5km[:,:]
        geo_1km[1][2::5, 2:-4:5] = lat_5km[:,:]
        #
        # Careful here: the scans overlap at large scan angles. Each scan is a band of 10 lines
        # of pixels. These lines do not overlap within the scan band - but the sequential bands do.
        # The 5km dots are the 3-rd and the 7-th lines
        # Note that along the scan all is fine: 5-cells jump is to be interpolated linearly
        # But along the track must account for 10-lines scan band 
        #
        a = np.array(range(0, 10, 2),dtype=np.float32) / 10.0   # linear interpolation coefficients
        a2 = np.array(range(-4,16,2), dtype=np.float32) / 10.0  # for restoring the whole band
        #
        # Sequentially applying the interpolation
        #
        for iG in [0,1]:
            # WRONG! Along the tract below is the correct way
#            for shift in range(1,5):
#                geo_1km[iG][2+shift:-3:5, 2:-4:5] = (geo_1km[iG][2:-3:5, 2:-4:5] * (1.-a[shift]) + 
#                                                     geo_1km[iG][7::5,   2:-4:5] * a[shift])
            # Along the track: account for 10-linebands
            for shift in range(10):
                geo_1km[iG][shift::10, 2:-4:5] = (geo_1km[iG][2::10, 2:-4:5] * (1.-a2[shift]) + 
                                                  geo_1km[iG][7::10,   2:-4:5] * a2[shift])
            # Along the scan, all is nice
            for shift in range(1,5):
                geo_1km[iG][2:, 2+shift:-7:5] = (geo_1km[iG][2:, 2:-9:5] * (1.-a[shift]) + 
                                                 geo_1km[iG][2:, 7:-4:5] * a[shift])

        # If we have a longitude jump, have to patch the region of -180 <-> 180
        if ifJump:
            lon_1km_tmp = np.zeros(shape=(lon_5km.shape[0]*5, lon_5km.shape[1]*5+4)) * np.nan
            # copy only two indices with jump in-between
            lon_1km_tmp[2::5, 2:-4:5][1:,:-1][idxJumpLon] = lon_5km[1:,:-1][idxJumpLon]
            lon_1km_tmp[2::5, 2:-4:5][:-1,:-1][idxJumpLon] = lon_5km[:-1,:-1][idxJumpLon]
            lon_1km_tmp[2::5, 2:-4:5][1:,1:][idxJumpLon] = lon_5km[1:,1:][idxJumpLon]
            lon_1km_tmp[2::5, 2:-4:5][:-1,1:][idxJumpLon] = lon_5km[:-1,1:][idxJumpLon]

            lon_1km_tmp[2::5, 2:-4:5][2:,:-1][idxJumpLon[:-1,:]] = lon_5km[2:,:-1][idxJumpLon[:-1,:]]
            lon_1km_tmp[2::5, 2:-4:5][:-2,:-1][idxJumpLon[1:,:]] = lon_5km[:-2,:-1][idxJumpLon[1:,:]]
            lon_1km_tmp[2::5, 2:-4:5][2:,1:][idxJumpLon[:-1,:]] = lon_5km[2:,1:][idxJumpLon[:-1,:]]
            lon_1km_tmp[2::5, 2:-4:5][:-2,1:][idxJumpLon[1:,:]] = lon_5km[:-2,1:][idxJumpLon[1:,:]]

            lon_1km_tmp[2::5, 2:-4:5][1:,:-2][idxJumpLon[:,1:]] = lon_5km[1:,:-2][idxJumpLon[:,1:]]
            lon_1km_tmp[2::5, 2:-4:5][:-1,:-2][idxJumpLon[:,1:]] = lon_5km[:-1,:-2][idxJumpLon[:,1:]]
            lon_1km_tmp[2::5, 2:-4:5][1:,2:][idxJumpLon[:,:-1]] = lon_5km[1:,2:][idxJumpLon[:,:-1]]
            lon_1km_tmp[2::5, 2:-4:5][:-1,2:][idxJumpLon[:,:-1]] = lon_5km[:-1,2:][idxJumpLon[:,:-1]]
            #
            # Within the copied region, turn to 0:360 from -180:180
#            idxFinite = np.isfinite(lon_1km_tmp)
#            lon1 = lon_1km_tmp[idxFinite]  # small array
#            print(np.nansum(lon_1km_tmp))
#            lon_1km_tmp[np.isfinite(lon_1km_tmp)][lon_1km_tmp[np.isfinite(lon_1km_tmp)] < 0] += 360.
#            print(np.nansum(lon_1km_tmp))
#            lon_1km_tmp[idxFinite][lon_1km_tmp[idxFinite] < 0] += 360.
#            print(np.nansum(lon_1km_tmp))
            lon_1km_tmp[lon_1km_tmp < 0] += 360.
#            print(np.nansum(lon_1km_tmp))
#            sys.exit()
            

#            print('\nPrefilled 1km with 5km map')
#            for i in range(lon_1km_tmp.shape[0]):
#                print( (' '.join('%6.2f' % lon_1km_tmp[i,jTmp] for jTmp in range(lon_1km_tmp.shape[1]))))
#
            #
            # Interpolate the region to 1km
            # Along the track: account for 10-linebands
            for shift in range(10):
                lon_1km_tmp[shift::10, 2:-4:5] = (lon_1km_tmp[2::10, 2:-4:5] * (1.-a2[shift]) + 
                                                  lon_1km_tmp[7::10,   2:-4:5] * a2[shift])
#            for shift in range(1,5):
#                lon_1km_tmp[2+shift:-3:5, 2:-4:5] = (lon_1km_tmp[2:-3:5, 2:-4:5] * (1.-a[shift]) + 
#                                                     lon_1km_tmp[7::5,   2:-4:5] * a[shift])
            for shift in range(1,5):
                lon_1km_tmp[2:, 2+shift:-7:5] = (lon_1km_tmp[2:, 2:-9:5] * (1.-a[shift]) + 
                                                 lon_1km_tmp[2:, 7:-4:5] * a[shift])
            #
            # Turn it back to -180 : 180
#            lon_1km_tmp[np.isfinite(lon_1km_tmp)][lon_1km_tmp[np.isfinite(lon_1km_tmp)] > 180] -= 360.
            lon_1km_tmp[lon_1km_tmp > 180] -= 360.

            
#            print('\nInterpolated jump region')
#            for i in range(lon_1km_tmp.shape[0]):
#                print( (' '.join('%6.2f' % lon_1km_tmp[i,jTmp] for jTmp in range(lon_1km_tmp.shape[1]))))
#            print('\nInterpolated 1km map before correction')
#            for i in range(lon_1km_tmp.shape[0]):
#                print( (' '.join('%6.2f' % geo_1km[0][i,jTmp] for jTmp in range(lon_1km_tmp.shape[1]))))
            #
            # Overwrite the region of the jump with correct values
            geo_1km[0][np.isfinite(lon_1km_tmp)] = lon_1km_tmp[np.isfinite(lon_1km_tmp)]

#            print('\nInterpolated 1km map after correction')
#            for i in range(lon_1km_tmp.shape[0]):
#                print( (' '.join('%6.2f' % geo_1km[0][i,jTmp] for jTmp in range(lon_1km_tmp.shape[1]))))

        # inner box is OK but outer bands, 2 cells wide from 3 sides, 6 cells wide from swath end
        # need extrapolation
        # Note that at -180 <-> 180 switch extrapolation may be not possible
        for iGeo in [0,1]:
            ny5km = lon_5km.shape[1]
            # start of ix. Cut y-wise ends, which are nans and cause warnings
            if np.any(np.abs(geo_1km[iGeo][2,2:ny5km*5-2] - 
                             geo_1km[iGeo][3,2:ny5km*5-2]) > 200.):
                geo_1km[iGeo][2:4,:][geo_1km[iGeo][2:4,:] < 0.] += 360.
            for i in [1, 0]:  
                geo_1km[iGeo][i,:] = 2. * geo_1km[iGeo][i+1,:] - geo_1km[iGeo][i+2,:]
            # end of ix
            nx = geo_1km[iGeo].shape[0]
            if np.any(np.abs(geo_1km[iGeo][nx-3,2:ny5km*5-2] - 
                             geo_1km[iGeo][nx-4,2:ny5km*5-2]) > 200.):
                geo_1km[iGeo][nx-4:nx-2,:][geo_1km[iGeo][nx-4:nx-2,:] < 0.] += 360.
            for i in range(nx-2, nx):
                geo_1km[iGeo][i,:] = 2. * geo_1km[iGeo][i-1,:] - geo_1km[iGeo][i-2,:]
            # start of iy
            if np.any(np.abs(geo_1km[iGeo][:,2] - geo_1km[iGeo][:,3]) > 200.):
                geo_1km[iGeo][:,2:4][geo_1km[iGeo][:,2:4] < 0.] += 360.
            for j in [1, 0]: 
                geo_1km[iGeo][:,j] = 2. * geo_1km[iGeo][:,j+1] - geo_1km[iGeo][:,j+2]
            # end of iy
            if np.any(np.abs(geo_1km[iGeo][:,ny5km*5-3] - geo_1km[iGeo][:,ny5km*5-4]) > 200.):
                geo_1km[iGeo][:,ny5km*5-4 : 
                              ny5km*5-1][geo_1km[iGeo][:,ny5km*5-4:ny5km*5-1] < 0.] += 360.
            for j in range(ny5km*5-2, geo_1km[iGeo].shape[1]):
                geo_1km[iGeo][:,j] = 2. * geo_1km[iGeo][:,j-1] - geo_1km[iGeo][:,j-2]
            # Basic stupidity: must be no nan-s
            if ifInputClean: 
                if not np.all(np.isfinite(geo_1km[iGeo])): 
                    print('1km grid has NOT been restored for ', {0:"lon", 1:'lat'}[iGeo])
                    xbad, ybad = np.where(np.logical_not(np.isfinite(geo_1km[iGeo]))) 
                    print(xbad.shape, ybad.shape)
                    for i in range(xbad.shape[0]):
                        print((xbad[i],ybad[i]))
            else: 
                self.log.log('Nans in input are ignored')

#        if ifJump:
#            print('\nInterpolated 1km map after correction and extrapolation')
#            for i in range(geo_1km[0].shape[0]):
#                print( (' '.join('%6.2f' % geo_1km[0][i,jTmp] for jTmp in range(geo_1km[0].shape[1]))))

#        # check that the field is smooth but keep in mind jumps around poles
#        for iGeo in [0,1]:
#            if (np.max(np.abs(geo_1km[iGeo][1:,:] - geo_1km[iGeo][:-1,:])) > 0.2 or
#                np.max(np.abs(geo_1km[iGeo][:,1:] - geo_1km[iGeo][:,:-1])) > 0.2):
#                if abs(geo_1km[1].flatten()[np.argmax(np.abs(geo_1km[iGeo][:,1:] - 
#                                                             geo_1km[iGeo][:,:-1]))]) < 70.0:
#                    self.log.log(self.satellite + ',  ' + str(self.now_UTC))
#                    self.log.log(self.templFRPFiles)
#                    self.log.log(self.templCloudFiles)
#                    self.log.log('Problematic smoothness axis %i step along i,j = %g,%g: lat: %g, %g' %
#                                 (iGeo, 
#                                  np.max(np.abs(geo_1km[iGeo][1:,:] - geo_1km[iGeo][:-1,:])), 
#                                  np.max(np.abs(geo_1km[iGeo][:,1:] - geo_1km[iGeo][:,:-1])),
#                                  geo_1km[1].flatten()[np.argmax(np.abs(geo_1km[iGeo][1:,:] - geo_1km[iGeo][:-1,:]))],
#                                  geo_1km[1].flatten()[np.argmax(np.abs(geo_1km[iGeo][:,1:] - geo_1km[iGeo][:,:-1]))]))
#                    # Print the results
#                    self.log.log('start 1km matrix corner:, axis %i' % iGeo)
#                    for i in range(15):
#                        self.log.log( ('  '.join('%7.3f' % geo_1km[iG][i,jTmp] for jTmp in range(15))))
#                    self.log.log('end 1km matrix corner:, axis %i' % iGeo)
#                    for i in range(15,0,-1):
#                        self.log.log( ('  '.join('%7.3f' % geo_1km[iG][-i,-jTmp] for jTmp in range(15,-1,-1))))
#                    self.log.log('\nFull longitude 5km field (after temporary roration)')
#                    for i in range(lon_5km.shape[0]):
#                        self.log.log( (' '.join('%6.2f' % lon_5km[i,jTmp] for jTmp in range(lon_5km.shape[1]))))
#                    self.log.log('\nFull latitude 5km field')
#                    for i in range(lon_5km.shape[0]):
#                        self.log.log( (' '.join('%6.2f' % lat_5km[i,jTmp] for jTmp in range(lon_5km.shape[1]))))
        # Storing
        self.lon = geo_1km[0]
        self.lon[self.lon > 180.] -= 360.
        self.lon[self.lon < -180.] += 360.
        self.lat = geo_1km[1]

#        # Print the results
#        print('LON  2')
#        for iG in [0,1]:
#            for i in range(15):
#                print( ('  '.join('%7.3f' % geo_1km[iG][i,jTmp] for jTmp in range(15))))
#            print('LAT  2')
#        print('LON  2')
#        for iG in [0,1]:
#            for i in range(15,0,-1):
#                print( ('  '.join('%7.3f' % geo_1km[iG][-i,-jTmp] for jTmp in range(15,-1,-1))))
#            print('LAT  2')
#        print('\n\n')
#        sys.exit()
        
        return geo_1km 


    #=============================================================

    def unpack_MxD03_byte_1(self, arIn):
        '''
        Bit fields within each byte are numbered from the left:
        7, 6, 5, 4, 3, 2, 1, 0.
        The left-most bit (bit 7) is the most significant bit.
        The right-most bit (bit 0) is the least significant bit.

         bit field       Description                             Key
         ---------       -----------                             ---
         0               Cloud Mask Flag                      0 = Not  determined
                                                              1 = Determined
         2, 1            Unobstructed FOV Quality Flag        00 = Cloudy
                                                              01 = Uncertain
                                                              10 = Probably  Clear
                                                              11 = Confident  Clear
                         PROCESSING PATH
                         ---------------
         3               Day or Night Path                    0 = Night  / 1 = Day
         4               Sunglint Path                        0 = Yes    / 1 = No
         5               Snow/Ice Background Path             0 = Yes    / 1 = No
         7, 6            Land or Water Path                   00 = Water
                                                              01 = Coastal
                                                              10 = Desert
                                                              11 = Land          '''
        # Create the object for storage
        #
        granBits = gb.unpacked_cloud_fields()
        granBits.ifCloudAnalysed = arIn[7,:,:]
        granBits.QA = arIn[6,:,:] + arIn[5,:,:] * 10
#        QA_txt_flg = np.where(QA_flg == 0, 'cloudy',
#                              np.where(QA_flg == 1, 'uncertain',
#                                       np.where(QA_flg == 10, 'clear_maybe',
#                                                np.where(QA_flg == 11, 'clear','ERROR'))))
        granBits.day_night = arIn[4,:,:]
        granBits.sunglint = arIn[3,:,:]
        granBits.snow = arIn[2,:,:]
        granBits.land = arIn[1,:,:] + arIn[0,:,:] * 10
#        land_txt_flg = np.where(land_flg == 0, 'water',
#                                np.where(land_flg == 1, 'coast',
#                                         np.where(land_flg == 10, 'desert',
#                                                  np.where(land_flg == 11, 'land','ERROR'))))
        granBits.QA_txt = {0 :'cloudy', 1 :'uncertain', 10 : 'clear_maybe',  11 : 'clear'}
        granBits.land_txt = {0 : 'water', 1 : 'coast', 10 : 'desert', 11 : 'land'}
        
        
        return granBits


    #=============================================================
    
    def unpack_MxD14_QA(self, arIn):
        '''
        Name                Type      Dimensions
        ----                ----      ----------
        algorithm QA        uint32    Dimension_1, Dimension_2
        
        Attribute                Type      Quantity  Value
        ---------                ----      --------  -----
        long_name                string         1    algorithm QA
        units                    string         1    bit field
        Nadir Data Resolution    string         1    1 km
        Bit       Description
        ---       -----------
        0-1       land/water state
                       00 = water,  01 = coast,  10 = land,  11 = UNUSED
        2         3.9 micron high-gain flag
                       0 = band 21 used,   1 = band 22 used
        3         atmospheric correction (0 = not performed, 1 = performed)
        4         day/night algorithm (0 = night, 1 = day)
        5         potential fire pixel
                       0 = no 1 = yes (if no, remaining QA fields except bit 23 are not set)
        6         spare (set to 0)
        7-10      background window size parameter, R
                       R = 0: unable to characterize background
                       R > 0: background characterized with (2R+1) by (2R+1) window
        11        360 K T21 test (0 = fail, 1 = pass)
        12        DT relative test i (0 = fail, 1 = pass)
        13        DT absolute test ii (0 = fail, 1 = pass)
        14        T21 relative test (0 = fail, 1 = pass)
        15        T31 relative test (0 = fail, 1 = pass)
        16        background fire T21 deviation test (0 = fail, 1 = pass)
        17-19     spare (set to 0)
        20        adjacent cloud pixel (valid only if fire mask = 7, 8, or 9)
                       0 = no,   1 = yes
        21        adjacent water pixel (valid only if fire mask = 7, 8, or 9)
                       0 = no,  1 = yes
        22-23     sun-glint level (0-3)
        24        sun glint rejection flag (0 = false, 1 = true)
        25        land-pixel desert boundary rejection flag (0 = false, 1 = true)
        26        land-pixel coastal false alarm rejection flag (0 = false, 1 = true)
        27        land-pixel forest clearing rejection test (0 = false, 1 = true)
        28        water-pixel coastal false alarm rejection flag (0 = false, 1 = true)
        29-31     spare (set to 0)
        Note: For coast pixels, bits 24-28 will have a value of 0.

        '''
        raise NotImplemented


    #=============================================================

    def pick_granule_data_IS4FIRES_v3_0(self, grid_to_fit_in=None):
        #
        # Following the above paradigm, we need:
        # - from MxD14:
        #      - FRP for fire pixels, their temperatures, locations in swath and lon-lat
        #      - cloud mask
        # - from MxD35:
        #      - longitude_reduced, latitude_reduced for geolocation
        #      - more detailed pixel properties: land, water, cloud, desert, day/night, QA
        #
        # First get the cloud file: it has geolocation
        #
        if self.now_UTC is None:   # a single file must be given, time will be taken from it
            arFNms = [self.templAuxilFiles]
        else:
            arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templAuxilFiles)))
        #
        # Open files but check for ambiguity
        #
        if len(arFNms) > 1:
            self.log.log('Several cloud files satisfy template:' + str(arFNms)) # warning
            return False
        elif len(arFNms) == 1:
            if ifDebug: self.log.log('Reading Cloud data from ' + os.path.split(arFNms[-1])[1])
            try:
                fIn = nc4.Dataset(arFNms[0],'r')  # netcdf file
                fIn.set_auto_maskandscale(False) ## Never mask, never scale
                ifHDF = False
                if 'longitude_reduced' in fIn.variables:
                    f1 = fIn.variables['longitude_reduced'][:,:]
                    f1_fill = fIn.variables['longitude_reduced']._FillValue
                    f2 = fIn.variables['latitude_reduced'][:,:]
                    f2_fill = fIn.variables['latitude_reduced']._FillValue
                else:
                    f1 = fIn.variables['Longitude'][:,:]
                    f1_fill = fIn.variables['Longitude']._FillValue
                    f2 = fIn.variables['Latitude'][:,:]
                    f2_fill = fIn.variables['Latitude']._FillValue
            except:
                try:
                    fIn = SD(arFNms[0])
                    ifHDF = True
                    f1 = fIn.select('Longitude')[:,:]
                    f1_fill = fIn.select('Longitude')._FillValue
                    f2 = fIn.select('Latitude')[:,:]
                    f2_fill = fIn.select('Latitude')._FillValue
                except:
                    self.log.log('FAILED FAILED FAILED: ' +arFNms[0])
                    return False
        else:
            self.log.log('No files for template:' + self.templAuxilFiles)
            return False
        #
        # check mask
        #
        try: lon5km = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f1.data)
        except: lon5km = f1
        lon5km[lon5km == f1_fill] = np.nan
        try: lat5km = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f2.data)
        except: lat5km = f2
        lat5km[lat5km == f2_fill] = np.nan
        #
        # If grid is given, can check that the granule is inside and return if not
        #
        if grid_to_fit_in is not None:
            fxSwath, fySwath = self.grid.geo_to_grid(lon5km, lat5km)
            ixSwath = np.round(fxSwath).astype(np.int)
            idxX_OK = np.logical_and(ixSwath >= 0, ixSwath < self.grid.nx)
            if not np.any(idxX_OK) : return False
            iySwath = np.round(fySwath).astype(np.int)
            idxY_OK = np.logical_and(iySwath >= 0, iySwath < self.grid.ny)
            if not np.any(idxY_OK) : return False
        #
        # Something is inside the grid. Proceed geolocation: 5 km and turn to 1km.
        #
        self.MODIS_1km_geo_from_5km(lon5km, lat5km)
        #
        # Get the cloud mask
        #
#        bits = np.unpackbits(fIn.variables['Cloud_Mask'][:,:,:].data, axis=0)
        # direct way fails, use this way - checked to work
        if ifHDF:
            cld_packed = fIn.select('Cloud_Mask')[:,:,:]
        else:
            cld_packed = fIn.variables['Cloud_Mask'][:,:,:]
        cld_bytes = np.zeros(shape=cld_packed.shape, dtype=np.uint8)  #'uint8')
        cld_bytes[:,:,:] = cld_packed[:,:,:]
        bits = np.unpackbits(cld_bytes,axis=0)
        #
        # Unpack the first byte 
        #
        self.BitFields = self.unpack_MxD03_byte_1(bits[:8,:,:])

        if ifHDF:
            fIn.end()
        else:
            fIn.close()
        #
        # Create a fire-noFire-unknown mask. For now, rules are (applied in this order):
        # - cloud and sunglint mean no-data
        # - land means zero for fires above threshold, no-data for smaller ones
        # - actual FRP means fire and its features
        # Coding:
        # np.nan for missing, negative threshold for conditional, actual FRP for fire
        #
        
        #
        # Clouds done, geolocation is known. Get FRP
        # FRP file has two parts: Fire-Pixel Table and packed maps - another set of bytes
        #
        if self.now_UTC is None:   # a single file must be given, time will be taken from it
            arFNms = [self.templFRPFiles]
        else:
            arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
        #
        # Open but still check for ambiguity
        #
        if len(arFNms) > 1: 
            self.log.log('Several FRP files satisfy template:' + str(arFNms))
            return False 
        elif len(arFNms) == 1:
            if ifDebug: print('Reading FRP from ', os.path.split(arFNms[0])[1])
            try:
                fIn = nc4.Dataset(arFNms[-1],'r')  # netcdf file
                fIn.set_auto_maskandscale(False) ## Never mask, never scale
                ifHDF = False
            except:
                fIn = SD(arFNms[0])
                ifHDF = True
        else:
            print('Template: ', self.templFRPFiles)
            print('Found files: ', glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
            return False
        #
        # Find time, if necessaary. Relying on the code metadata attributes
        # A very tedious task: the attribute is a long string packing the structure. In particular:
        # for date (sTmp2 below):
        # ..... 'RANGEBEGINNINGDATE\n      NUM_VAL              = 1\n      VALUE                = "2024-01-01"\n    END_OBJECT             = '  ......
        # for time (sTmp3 below):
        # .....  'RANGEBEGINNINGTIME\n      NUM_VAL              = 1\n      VALUE                = "00:00:00.000000"\n    END_OBJECT             = '  ......
        # The file name corresponds to the beginning of the range, so take this one
        #
        if self.now_UTC is None:
            if ifHDF:
                sTmp = fIn.attributes()['CoreMetadata.0']
            else:
                sTmp = fIn.getncattr('CoreMetadata.0')
            # Start time of the granule:
            sTmp2 = sTmp[sTmp.find('RANGEBEGINNINGDATE'):sTmp.rfind('RANGEBEGINNINGDATE')]
            sTmp3 = sTmp[sTmp.find('RANGEBEGINNINGTIME'):sTmp.rfind('RANGEBEGINNINGTIME')]
            try:
                self.tBegin = dt.datetime.strptime(sTmp2.split('"')[1] + '__' + sTmp3.split('"')[1].split('.')[0],
                                                   '%Y-%m-%d__%H:%M:%S')
            except:
                print(sTmp[sTmp.find('RANGEBEGINNINGDATE'):sTmp.rfind('RANGEBEGINNINGDATE')])
                print(sTmp3 = sTmp[sTmp.find('RANGEBEGINNINGTIME'):sTmp.rfind('RANGEBEGINNINGTIME')])
                raise ValueError('failed to decode time from attributes:')
            # End time of the granule
            sTmp2 = sTmp[sTmp.find('RANGEENDINGDATE'):sTmp.rfind('RANGEENDINGDATE')]
            sTmp3 = sTmp[sTmp.find('RANGEENDINGTIME'):sTmp.rfind('RANGEENDINGTIME')]
            try:
                self.tEnd = dt.datetime.strptime(sTmp2.split('"')[1] + '__' + sTmp3.split('"')[1].split('.')[0],
                                                 '%Y-%m-%d__%H:%M:%S')
            except:
                print(sTmp[sTmp.find('RANGEBEGINNINGDATE'):sTmp.rfind('RANGEBEGINNINGDATE')])
                print(sTmp3 = sTmp[sTmp.find('RANGEBEGINNINGTIME'):sTmp.rfind('RANGEBEGINNINGTIME')])
                raise ValueError('failed to decode time from attributes:')
            
            self.now_UTC = self.tBegin #+ 0.5 * (self.tEnd - self.tBegin)   # midpoint of the granule seems reasonable here
        #
        # Get the basic information on the fires - same set as for IS4FIRES v.2.0
        #
        if ifHDF:
            self.nFires = fIn.datasets()['FP_power'][1][0]
            if self.nFires == 0:   # dimension
                if ifDebug: self.log.log('No fires 1, %s' % str(self.now_UTC))
                fIn.end()
                return True
            else:
                if ifDebug: self.log.log('Nbr of fires debug: %g, %s'  % (self.nFires, str(self.now_UTC)))
                FP_frp = fIn.select('FP_power')[:]
        else:
            try:
                FP_frp = fIn.variables['FP_power'][:]
            except:
                if ifDebug: self.log.log('No fires 2, %s' % str(self.now_UTC))
                self.nFires = 0
                fIn.close()
                return True
        self.nFires = FP_frp.shape[0]
        self.log.log('Nbr of fires: %i, %s' % (self.nFires, str(self.now_UTC)))
        #
        # Rest will continue with the fire records cless
        #
        self.FP = fire_records.fire_records(self.log)
        try:
            self.FP.init_data_structures(self.nFires)
        except:
            print(self.nFires)
            self.FP.init_data_structures(self.nFires)
        #
        # If fires exist, fill in
        #
        if ifHDF:
            self.FP.FRP = fIn.select('FP_power')[:]
            self.FP.lon = fIn.select('FP_longitude')[:]
            self.FP.lat = fIn.select('FP_latitude')[:]
            self.FP.line = fIn.select('FP_line')[:]
            self.FP.sample = fIn.select('FP_sample')[:]
            self.FP.dS = self.dS[self.FP.sample]   # depends on swath 
            self.FP.dT = self.dT[self.FP.sample]   # also depends on swath
            self.FP.T4 = fIn.select('FP_T21')[:]
            self.FP.T4b = fIn.select('FP_MeanT21')[:]
            self.FP.T11 = fIn.select('FP_T31')[:]
            self.FP.T11b = fIn.select('FP_MeanT31')[:]
            self.FP.TA = fIn.select('FP_MeanDT')[:]
            self.FP.SolZenAng = fIn.select('FP_SolZenAng')[:]
            self.FP.ViewZenAng = fIn.select('FP_ViewZenAng')[:]
            fIn.end()
        else:
            self.FP.FRP = fIn.variables['FP_power'][:]
            self.FP.lon = fIn.variables['FP_longitude'][:]
            self.FP.lat = fIn.variables['FP_latitude'][:]
            self.FP.line = fIn.variables['FP_line'][:]
            self.FP.sample = fIn.variables['FP_sample'][:]
            self.FP.dS = self.dS[self.FP.sample]   # depends on swath 
            self.FP.dT = self.dT[self.FP.sample]   # also depends on swath
            self.FP.T4 = fIn.variables['FP_T21'][:]
            self.FP.T4b = fIn.variables['FP_MeanT21'][:]
            self.FP.T11 = fIn.variables['FP_T31'][:]
            self.FP.T11b = fIn.variables['FP_MeanT31'][:]
            self.FP.TA = fIn.variables['FP_MeanDT'][:]
            self.FP.SolZenAng = fIn.variables['FP_SolZenAng'][:]
            self.FP.ViewZenAng = fIn.variables['FP_ViewZenAng'][:]
            fIn.close()
        self.FP.satellite = np.array([self.abbrev_1(arFNms[-1])] * self.nFires, dtype='|S1')
        self.FP.QA_flag = 0
        self.FP.QA_msg = ''
        self.FP.timezone = 'UTC'
        self.FP.timeStart = self.now_UTC
        self.FP.LU_metadata = ''
        self.FP.nFires = self.nFires
        return True


#    #=============================================================
#
#    def detection_limit(self):
#        #
#        # returns the detection limit map as a function of the day/night flag and pixel size
#        # Coefficients for the regression pixel - detection_limit are determined as the linear
#        # regression of the 1%-tile of FRP distribution over 2000-2019, both Terra and Aqua
#        # Whole-day regression gives slope=5.52 and intercept 0.66.
#        # year to year standard deviation is 
#        # If taking years 2000-2020 separately and then averaging:
#        # Aday_mean, Bday_mean, Anight_mean, Bnight_mean, Aday_std, Bday_std, Anight_std, Bnight_std
#        # 3.645      0.694      3.363        0.389        0.0594   0.1365      0.050       0.167
#        #
#        self.slope_day = 3.53
#        self.intercept_day_MW = 0.80
#        self.slope_night = 3.36
#        self.intercept_night_MW = 0.37
#        return np.where(self.BitFields.day_night == 1, 
#                        self.slope_day * self.area[None,:] + self.intercept_day_MW,
#                        self.slope_night * self.area[None,:] + self.intercept_night_MW
#                        ).reshape(self.BitFields.day_night.shape).astype(np.float32)
        
        
    #=============================================================

    def pick_granule_data_IS4FIRES_v2_0(self):
        #
        # Following the above paradigm, we need:
        # - from MxD14:
        #      - FRP for fire pixels, their temperatures, locations in swath and lon-lat
        #      - cloud mask
        # - from MxD35:
        #      - longitude_reduced, latitude_reduced for geolocation
        #      - more detailed pixel properties: land, water, cloud, desert, day/night, QA
        #
        # Get FRP
        # FRP file has two parts: Fire-Pixel Table and packed maps - another set of bytes
        #
        arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
        if len(arFNms) == 1:
            print('Reading FRP from ' + arFNms[0])
            fIn = nc4.Dataset(arFNms[0], 'r')
        elif len(arFNms) > 1:
            self.log.log('Several FRP files satisfy template:' + str(arFNms))
            return False
        else:
            print('Template:', self.templFRPFiles, self.now_UTC)
            print('Found files: ', glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
            return False
        #
        # Get the IS4FIRES v.2.0 variables: FRP and ix, iy in the swath
        #
        FP_frp = fIn.variables['FP_power'][:]
        self.nFires = FP_frp.shape[0] 
        if len(FP_frp) == 0: return True         # empty granule
        # fire_records class
        self.FP = fire_records.fire_records(self.log)
        self.FP.init_data_structures(self, self.nFires)
        self.FP.FRP = fIn.variables['FP_power'][:]
        self.FP.line = fIn.variables['FP_line'][:]     # probably index along the track [0:2029]
        self.FP.sample = fIn.variables['FP_sample'][:] # probably, along the swath, [0:1353]
        self.FP.lon = fIn.variables['FP_longitude'][:]
        self.FP_lat = fIn.variables['FP_latitude'][:]
        print(np.min(self.FP.line), np.max(self.FP.line), 
              np.min(self.FP.sample), np.max(self.FP.sample))
        self.FP.dS = self.dS[self.FP_sample]
        self.FP.dT = self.dT[self.FP_sample]
        self.FP.T4 = fIn.variables['FP_T21'][:]
        self.FP.T4b = fIn.variables['FP_MeanT21'][:]
        self.FP.T11 = fIn.variables['FP_T31'][:]
        self.FP.T11b = fIn.variables['FP_MeanT31'][:]
        self.FP.TA = fIn.variables['FP_MeanDT'][:]
        self.FP.SolZenAng = fIn.variables['FP_SolZenAng']
        self.FP.satellite = np.array([self.abbrev_1(arFNms[0])]*self.nFires, dtype='|S1')
        self.FP.time = np.zeros(shape=(self.nFires),dtype=np.int64)
        self.FP.QA_flag = 0
        self.FP.QA_msg = ''
        self.FP.timezone = 'UTC'
        self.FP.LU_metadata = ''
        self.FP.timeStart = self.now_UTC
        return True


    #===================================================================
    
    def write_granule_IS4FIRES_v2_0(self, iStartFire, fOut):
        # write it down
        # fireNbr yr mon day hr min sec lon lat dS dT km frp MW T4 T4b T11 T11b TA MCE FireArea
        for iFire in range(self.nFires):
            fOut.write('fire = %03i %s %g %g %g %g km %g MW %g %g %g %g %g %g %g %g %s\n' %
                       (iFire + iStartFire, self.now_UTC.strftime('%Y %m %d %H %M 0.0'), 
                        self.FP.lon[iFire], self.FP.lat[iFire], self.FP.dS[iFire], self.FP.dT[iFire],
                        self.FP.FRP[iFire], self.FP.T4[iFire], self.FP.T4b[iFire], self.FP.T11[iFire], 
                        self.FP.T11b[iFire], self.FP.TA[iFire], 0.5, 0.01,   # FP_MCE[iFire], FP_FireArea[iFire))
                        self.FP.SolZenAng[iFire], self.FP.satellite[iFire]))
# SILAm reads like this:
#        strTmp, strTmp1, iFire, yr, mon, day, hr, mn, sec, &
#                       & fLonTmp, fLatTmp, &
#                       & fDx, fDy, chSizeUnit, fFRP, chFRPUnit, &
#                       & fpT4, fpT4b, fpT11, fPT11b, fpTA, fpMCE, fParea
        return iStartFire + self.FP_frp.shape[0]


    #===================================================================
    
    def to_nc(self, chOut_FNm_or_handle):
        #
        # Stores the granule to the netCDF file. 
        #
        # file name or handler?
        if type(chOut_FNm_or_handle) is str:
            outF = nc4.Dataset(chOut_FNm_or_handle , "w", format="NETCDF4")
        else:
            outF = chOut_FNm_or_handle

        outF.featureType = "MODIS_granule";
        time_unit = self.now_UTC.strftime("seconds since %Y-%m-%d %H:%M:%S UTC")
        outF.time = netcdftime.date2num(self.now_UTC, units=time_unit, calendar='standard')
        outF.time_unit = time_unit
        outF.FP_timezone = 'UTC'
        outF.nFires = self.nFires
        if 'MOD14' in self.templFRPFiles: outF.satellite = 'O'
        elif 'MYD14' in self.templFRPFiles: outF.satellite = 'Y'
        else: raise ValueError('Cannot get satellite from template: ' + self.templFRPFiles)
        #
        # The dimensions: line, sample, and, possibly, fires that are set by fire_records
        #
        lineAxis = outF.createDimension("line", self.lon.shape[0])   # number of lines along the track
        sampleAxis = outF.createDimension("sample", self.lon.shape[1])  # == self.N, number of pixels in a scan line
        # 
        # Store variables
        # Maps
        #
        for map_var in [('lon','f4','longitude','degrees_east', self.lon),   # var_name, type, long name, unit
                        ('lat','f4','latitude','degrees_north', self.lat),
                        ('cld_mask','i1','Cloud mask(1=made)', '', self.BitFields.ifCloudAnalysed),
                        ('cld_clr', 'i1', 'cloud-clear: 0=cloud,1=uncertain,10=clear?,11=clear','',self.BitFields.QA),
                        ('d_n','i1', '1=day, 0=night','',self.BitFields.day_night), 
                        ('sun_gl','i1','0=sunglint','',self.BitFields.sunglint),
                        ('snow','i1','0=snow','',self.BitFields.snow),
                        ('water_land','i1','0=water,1=coast,10=desert,11=land','',self.BitFields.land)]:
            vMap = outF.createVariable(map_var[0], map_var[1], ("line","sample"), zlib=True, complevel=5)
#                                              least_significant_digit=5)
            vMap.long_name = map_var[2]
            #if map_var[3] != '': 
            vMap.units = map_var[3]
            outF.variables[map_var[0]][:,:] = map_var[4][:,:]
        #
        # Fires
        #
        if self.nFires > 0:
            self.FP.to_nc(outF)
#            for FP_var in [('FP_frp','f4','FRP','MW', self.FP_frp),   # var_name, type, long name, unit
#                           ('FP_lon','f4','longitude','degrees_east', self.FP_lon),
#                           ('FP_lat','f4','latitude','degrees_north', self.FP_lat),
#                           ('FP_T4','f4','temperature 3.96 um, T21','K', self.FP_T4),
#                           ('FP_T4b','f4','temperature background 3.96 um, MeanT21','K', self.FP_T4b),
#                           ('FP_T11','f4','temperature 11 um, T31','K', self.FP_T11),
#                           ('FP_T11b','f4','temperature background 11 um, meanT31','K', self.FP_T11b),
#                           ('FP_line','i4','granule line of fire pixel','', self.FP_line),
#                           ('FP_sample','i4','granule sample of fire pixel','', self.FP_sample),
#                           ('FP_SolZenAng','f4','solar zenith angle','degrees', self.FP_SolZenAng),
#                           ('FP_ViewZenAng','f4','view zenith angle','degrees', self.FP_ViewZenAng)]:
#                vFP = outF.createVariable(FP_var[0], FP_var[1], ("fires"), zlib=True, complevel=5)
#    #                                          least_significant_digit=5)
#                vFP.long_name = FP_var[2]
#                if FP_var[3] != '': vFP.units = FP_var[3]
#                outF.variables[FP_var[0]][:] = FP_var[4][:self.nFires]  # shapes of arrays can be larger

        if type(chOut_FNm_or_handle) is str:
            outF.close()
        

    #===================================================================
    
    def from_nc(self, chIn_FNm_or_handle):
        #
        # Stores the granule to the netCDF file. 
        #
        # file name or handler?
        if type(chIn_FNm_or_handle) is str:
            fIn = nc4.Dataset(chIn_FNm_or_handle , "r")
        else:
            fIn = chIn_FNm_or_handle

        self.now_UTC = netcdftime.num2date(fIn.time, fIn.time_unit)
#        startTime = dt.datetime.strptime(fIn.time_unit, 'seconds since %Y-%m-%d %H:%M:%S UTC')
#        self.now_UTC = startTime + spp.one_second * fIn.time
        self.nFires = fIn.nFires
        
#        if 'MOD14' in self.templFRPFiles: outF.satellite = 'O'
#        elif 'MYD14' in self.templFRPFiles: outF.satellite = 'Y'
#        else: raise ValueError('Cannot get satellite from template: ' + self.templFRPFiles)
        # 
        # Read variables
        # Maps
        #
        self.lon = fIn.variables['lon'][:,:].astype(np.float32)
        self.lat = fIn.variables['lat'][:,:].astype(np.float32)
        self.BitFields = gb.unpacked_cloud_fields()
        self.BitFields.ifCloudAnalysed = fIn.variables['cld_mask'][:,:].astype(np.int8)
        self.BitFields.QA = fIn.variables['cld_clr'][:,:].astype(np.int8)
        self.BitFields.day_night = fIn.variables['d_n'][:,:].astype(np.int8)
        self.BitFields.sunglint = fIn.variables['sun_gl'][:,:].astype(np.int8)
        self.BitFields.snow = fIn.variables['snow'][:,:].astype(np.int8)
        self.BitFields.land = fIn.variables['water_land'][:,:].astype(np.int8)
        #
        # Fires
        #
        if self.nFires > 0:
            self.FP = fire_records.fire_records(self.log)
            self.FP.from_nc(fIn)

        if type(chIn_FNm_or_handle) is str:
            fIn.close()

    #===================================================================================    
        
    def make_fire_records(self, inFRP, inLine, inSample):
        #
        # If FRP is given together with its indices in teh granule, fire records can be created 
        # with a few missing values
        #
        self.FP = fire_records.fire_records(self.log)
        self.nFires = len(inFRP)
        self.FP.from_dic({'lon': self.lon[inLine,inSample], 'lat': self.lat[inLine,inSample],
                          'time': np.zeros(shape=(self.nFires)), 
                          'FRP': inFRP,
                          'dS':self.dS[inSample], 'dT':self.dT[inSample], 
                          'sza':spp.solar_zenith_angle(self.lon[inLine,inSample], 
                                                       self.lat[inLine,inSample], 
                                                       self.now_UTC.timetuple().tm_yday,
                                                       self.now_UTC.hour, self.now_UTC.minute),
                          'ViewZenAng': self.theta[inSample],
                          'T4':np.ones(shape=(self.nFires))*(-1), 'T4b':np.ones(shape=(self.nFires))*(-1),
                          'T11':np.ones(shape=(self.nFires))*(-1), 'T11b':np.ones(shape=(self.nFires))*(-1),
                          'TA':np.ones(shape=(self.nFires))*(-1), 
                          'ix':np.ones(shape=(self.nFires))*(-1), 'iy':np.ones(shape=(self.nFires))*(-1),
                          'i_line':inLine, 'i_sample':inSample, 
                          'LU': np.ones(shape=(self.nFires))*(-1), 'satellite':'M',
                          'grid':None, 'gridName':None, 'timezone':'UTC',
                          'land_use_metadata':'',
                          'timeStart':self.now_UTC,
                          'QA_flag':0})
    
    
    #===================================================================

    def draw_granule(self, chOutFNm):
        #
        # Area covered by this swath, with a bit free space around:
        #
        # This is to draw a few cross-sections along the scan, just 50 point
#        figTmp, ax = mpl.pyplot.subplots(1,1, figsize=(16,10))
#        for shift in range(0,self.lat_1km.shape[1],100):
#            pltLat = mpl.pyplot.plot(range(50), self.lat_1km[:50,shift], marker='.')
#        mpl.pyplot.show()
#        sys.exit()
        
        fig, axes = mpl.pyplot.subplots(4,2, figsize=(10,16))
        minLon = np.nanmin(self.lon)
        minLat = np.nanmin(self.lat)
        maxLon = np.nanmax(self.lon)
        maxLat = np.nanmax(self.lat)
        cmap = mpl.pyplot.get_cmap('cool')
        ixAx = 0
        iyAx = 0
        chFires = ' no fires'
        for var, chTxt, cm in [(self.BitFields.ifCloudAnalysed,'Cloud mask(1=made) + FRP, %i MW','cool'),
                               (self.BitFields.QA,'QA: 0=cld,1=?,10=clr?,11=clr','Paired'),
                               (self.BitFields.day_night, '1=day, 0=night','cool'), 
                               (self.BitFields.sunglint,'0=sunglint','cool'),
                               (self.BitFields.snow,'0=snow','cool'),
                               (self.BitFields.land,'0=water,1=coast,10=desert,11=land','Paired'),
                               (self.lon,'Lon','cool'),
                               (self.lat,'Lat','cool')]:
            ax = axes[ixAx,iyAx]
#            print('Plotting ', chTxt)
            # draw coastlines, state and country boundaries, edge of map.
            bmap = basemap.Basemap(projection='cyl', resolution='l',    # crude, low, intermediate, high, full 
                                   llcrnrlon = minLon - 1.5, urcrnrlat = min(90.,maxLat + 1.5),
                                   urcrnrlon = maxLon + 1.5, llcrnrlat = max(-90.,minLat - 1.5),
                                   ax=ax)
            bmap.drawcoastlines(linewidth=0.5)
            bmap.drawcountries(linewidth=0.4)
            # draw parallels and meridians
            bmap.drawmeridians(np.arange(np.round(minLon - 0.2), np.round(maxLon + 0.2), 
                                         np.round((maxLon - minLon + 0.4) / 5.0)),
                               labels=[0,0,0,1],fontsize=10)
            bmap.drawparallels(np.arange(np.round(max(-90.,minLat - 0.2)), 
                                         np.round(min(90.,maxLat + 0.2)), 
                                         np.round((maxLat - minLat + 0.4) /5.0)), 
                               labels=[1,0,0,0],fontsize=10)
            # draw filled contours.
            #        cs = bmap.imshow(var.T, norm=None, cmap=cmap)
            cs = bmap.scatter(self.lon[::2,::2].T, self.lat[::2,::2].T, c=var[::2,::2].T, 
                              s=1, edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap(cm))
            cbar = bmap.colorbar(cs,location='bottom',pad="7%")
#            cbar.set_label(chTxt, fontsize=9)
            
            if ixAx + iyAx == 0:
                if self.nFires>0:  #FFP_frp.shape[0] > 0:
#                    print('Plotting FRP')
                    chFires = ', %g fires' % self.nFires
                    sort_order = self.FP.FRP.argsort()
                    cs2 = bmap.scatter(self.FP.lon[sort_order], self.FP.lat[sort_order], 
                                       c=self.FP.FRP[sort_order], s=30, 
                                       edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap('rainbow'))
                    cbar2 = bmap.colorbar(cs2,location='right',pad="7%")
                    cbar2.set_label('FRP, MW', fontsize=9)
                    ax.set_title(chTxt % np.round(np.sum(self.FP.FRP)).astype(int), fontsize=10)
                else: ax.set_title(chTxt % 0, fontsize=10)
            else: ax.set_title(chTxt, fontsize=10)
            
            if ixAx == 3:
                ixAx = 0
                iyAx += 1
            else: ixAx += 1

        # If FRP added, remove labels on the right-hand-side map
        if self.nFires > 0: #FP_frp.shape[0] > 0:
            axes[0,1].yaxis.set_ticklabels([])
        mpl.pyplot.suptitle(os.path.split(chOutFNm)[1].strip('.png') + chFires, fontsize=14)
        mpl.pyplot.savefig(chOutFNm,dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()


##################################################################################

def check_pixel_fit(chLUT_FNm):
    # Initial grid cell sizes: half of the scan, starting from nadir
    with open(chLUT_FNm,'r') as fIn:
        i_pix = []
        dS_pix = []
        dT_pix = []
        thet_pix = []
        for line in fIn:
            if line.startswith('#'): continue
            flds = line.split(';')
            if int(flds[0]) > 1354 //2: break  # somehow, the file includes more pixels than MODIS has
            i_pix.append(int(flds[0]))
            dS_pix.append(float(flds[8]))
            dT_pix.append(float(flds[9]))
            thet_pix.append(float(flds[3]))
            if abs(dS_pix[-1] * dT_pix[-1] / float(flds[10]) - 1) > 1e-3:
                print('Wrong line', line)
                sys.exit()
    dS_LUT = np.array(dS_pix[:-1])
    dT_LUT = np.array(dT_pix[:-1])
    t = np.array(thet_pix)
    theta_LUT = (t[1:] + t[:-1])/2
    # get full scan
    dS_LUT = np.concatenate((dS_LUT[::-1],dS_LUT))
    dT_LUT = np.concatenate((dT_LUT[::-1],dT_LUT))
    theta_LUT = np.concatenate((theta_LUT[::-1],theta_LUT))

    # ATBD of MODIS
    gran = granule_MODIS('MODIS', dt.datetime.utcnow(), '', '', 
                         spp.log(os.path.join(dirMain,'fit_pixel_size.log')))
    gran.log.log('\ndS: i, dS_LUT, dS_fla, DS_LUT-dS_fla, rel_error')
    for i in range(gran.N):
        if (np.abs(dS_LUT[i] - gran.dS[i]) / dS_LUT[i] > 0.01 
            or np.abs(dT_LUT[i] - gran.dT[i]) / dT_LUT[i] > 0.01): 
            gran.log.log('%i %g %g %g %g      %g %g %g %g <<<<<<=============' % 
                         (i, dS_LUT[i], gran.dS[i], dS_LUT[i] - gran.dS[i], 
                          np.abs(dS_LUT[i] - gran.dS[i]) / dS_LUT[i],
                          dT_LUT[i], gran.dT[i], dT_LUT[i] - gran.dT[i], 
                          np.abs(dT_LUT[i] - gran.dT[i]) / dT_LUT[i]))
        else:
            gran.log.log('%i %g %g %g %g      %g %g %g %g' % 
                         (i, dS_LUT[i], gran.dS[i], dS_LUT[i] - gran.dS[i], 
                          np.abs(dS_LUT[i] - gran.dS[i]) / dS_LUT[i],
                          dT_LUT[i], gran.dT[i], dT_LUT[i] - gran.dT[i], 
                          np.abs(dT_LUT[i] - gran.dT[i]) / dT_LUT[i]))
        if (np.abs(dS_LUT[i] - gran.dS[i]) / dS_LUT[i] > 0.1 
            or np.abs(dT_LUT[i] - gran.dT[i]) / dT_LUT[i] > 0.1): 
            gran.log.log('\n\n\n**************************************\n\n\n')
    # Summary
    gran.log.log('dS error: min_LUT-fla, max_LUT-fla, max_relative: %g %g %g' %
                 (np.min(dS_LUT - gran.dS), np.max(dS_LUT - gran.dS),
                  np.max(np.abs(dS_LUT - gran.dS)/dS_LUT)))
    gran.log.log('dT error: min_LUT-fla, max_LUT-fla, max_relative: %g %g %g' %
                 (np.min(dT_LUT - gran.dT), np.max(dT_LUT - gran.dT),
                  np.max(np.abs(dT_LUT - gran.dT)/dT_LUT)))
    Th1 = theta_LUT * np.pi / 180.0
    dNadirPixel = 0.3841 / gran.h
    gran.log.log('theta error: min_LUT-fla, max_LUT-fla, max_relative: %g %g %g' %
                 (np.min(Th1 - np.abs(gran.theta)), np.max(Th1 - np.abs(gran.theta)),
                  np.max(np.abs(Th1 - np.abs(gran.theta))/dNadirPixel)))
#        for i in range(gran.theta.size):
#            gran.log.log('%i %g' % (i, gran.theta[i]))
        
    # Draw the difference
    fig, axes = mpl.pyplot.subplots(3,1, figsize=(6, 12))
    # dS
    ax1 = axes[0].twinx()
    axes[0].plot(range(gran.N), gran.dS, label='dS', c='blue') #, dS_LUT - gran.dS)
    ax1.plot(range(gran.N), dS_LUT - gran.dS, label='diff', c='red')
    axes[0].set_ylabel('dS, km', color='blue')
    ax1.set_ylabel('dS_LUT - dS, km', color='red')
    # dT
    ax2 = axes[1].twinx()
    axes[1].plot(range(gran.N), gran.dT, label='dT', c='blue') #, dS_LUT - gran.dS)
#        axes[1].plot(range(gran.N), dT_LUT+0.1, label='dT_LUT', c='green') #, dS_LUT - gran.dS)
    ax2.plot(range(gran.N), dT_LUT - gran.dT, label='diff_dT', c='red')
    axes[1].set_ylabel('dT, km', color='blue')
    ax2.set_ylabel('dT_LUT - dT, km', color='red')
        # theta
    ax3 = axes[2].twinx()
    axes[2].plot(range(gran.N), gran.theta, label='theta', c='blue') #, dS_LUT - gran.dS)
    ax3.plot(range(gran.N), theta_LUT * np.pi / 180.0 - np.abs(gran.theta), label='diff', c='red')
    axes[2].set_ylabel('theta, rad', color='blue')
    ax3.set_ylabel('theta_LUT - theta, rad', color='red')
    mpl.pyplot.savefig(os.path.join(dirMain,'pixel_size_fit.png'), bbox_inches='tight', dpi=200)
#        mpl.pyplot.show()



##########################################################################
#
# Open-up the dS, dT, and theta for external use without creating granule
#
granMODIS_ref = granule_MODIS()
dS = granMODIS_ref.dS
dT = granMODIS_ref.dT
theta = granMODIS_ref.theta


##########################################################################

if __name__ == '__main__':
    
    dirMain = 'd:\\data'
    
    ifGeometry = True
    ifTestGranule = False
    ifDrawGranule = False
    ifCheckSizeFit = False
    #
    # Draw the basic features of the satellite geometry
    #
    if ifGeometry:
        granule = granule_MODIS(now_UTC=dt.datetime(2023,3,5,8,55))
        granule.draw_pixel_size(os.path.join(dirMain, 'MODIS_pixel.png'),
                                ifOverlap_map=True) 
    #
    # test the MODIS data
    #
    if ifTestGranule:
        gv = granule_MODIS('TERRA', dt.datetime(2022,5,8),
                           'd:\\data\\VIIRS\\VNP14\\2022.05.08\\VNP14IMG.A2022128.0000.001.2022128065445.nc', 
                           'd:\\data\\VIIRS\\VNP03IMGLL\\2022.05.08\\VNP03IMGLL.A2022128.0000.001.2022128063411.h5', 
                           spp.log('d:\\tmp\\MODIS_tst.log'))
        print(gv.type)
        gv.draw_pixel_size('d:\\tmp\\MODIS_pixel.png')
    #
    # draw a bunch of granules
    #
    if ifDrawGranule:
        for i in range(int(24*12)):
            t = dt.datetime(2020,8,7) + i * spp.one_minute
    #    for chFNm in glob.glob('d:\\data\\MODIS_extract\\2020\\220\\20200807\\*'):
            gv = granule_MODIS('TERRA', t,
                               os.path.join('d:\\data\\MODIS\\MOD14_coll_6\\%Y\\%Y.%m.%d','MOD14_L2.A%Y%j.%H%M.006.*.hdf'),
                               os.path.join('d:\\data\\MODIS\\MOD35_coll_61\\%Y\\%j','MOD35_L2.A%Y%j.%H%M.061.*.hdf'),
                               spp.log('d:\\tmp\\MODIS_tst.log'))
            gv.pick_granule_data_IS4FIRES_v3_0()
            gv.draw_granule('d:\\tmp\\' + t.strftime('gran_%Y%m%d_%H%M.png'))

    if ifCheckSizeFit:
        check_pixel_fit('d:\\data\\satellites\\graph_pixel_size_MODIS.csv')        

