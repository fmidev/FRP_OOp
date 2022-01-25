'''
 The class holding the information on a single swatch of the MODIS satellite 

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

Functions available:
- __init__:   Sets abasic file names 
- get_MODIS_pixel_size:  basic MODIS geomatry
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

import numpy as np
import glob
from os import path
import matplotlib as mpl
import netCDF4 as nc4
from mpl_toolkits import basemap


ifDebug = False


#################################################################################
#
#  A void class made for storing named fields after unpacking them from the 
#  granule bit-pacjed variables
#
#################################################################################

class unpacked_cloud_fields():
    def __init(self):
        return


#################################################################################
#
# The class holding the information on a single swatch of the satellite, 
# ~ 2030 x 1354 pixels and 5 min of the orbit for MODIS
#
#################################################################################

class MODIS_granule():

    #=============================================================
    def __init__(self, satellite, now_UTC, chFRPfilesTempl, chCloudFilesTempl, log):
        self.satellite = satellite
        self.now_UTC = now_UTC
        self.templFRPFiles = chFRPfilesTempl        # ready for now.strftime(template)
        self.templCloudFiles = chCloudFilesTempl
        self.get_MODIS_pixel_size()        # get the distribution of MODIS swath
        self.log = log


    #=============================================================

    def get_MODIS_pixel_size(self):
        #
        # An approximate but quite accurate formula for dS and dT pixel size along the scan 
        # and along the track directions, respectively
        # Presented in Ichoku & Kaufman, IEEE Transactions 2005
        #
        R_Earth = 6378.137  # Earth radius, km
        h_MODIS = 705.      # MODIS altitude, km
        N_MODIS = 1354      # number of pixels in a swatch row
        h = h_MODIS
        s = 0.0014184397  # ratio of nadir pixel size to h
        r = R_Earth + h
        theta = np.array( list( ((s * (0.5*(-N_MODIS+1) + i)) for i in range(N_MODIS)) ) )
        sinTmp = np.sin(theta) 
        cosTmp = np.cos(theta)
        aTmp = np.sqrt((R_Earth/ r)**2 - np.square(sinTmp))
        # distance from swatch centre point
        c = r * cosTmp - np.sqrt( r*r* np.square(cosTmp) + R_Earth*R_Earth - r*r)
        dd = R_Earth * np.arcsin(c * sinTmp / R_Earth)
        # pixel sizes
        self.dS = R_Earth * s * ((cosTmp / aTmp) - 1.0)   # size along swath
        self.dT = r * s * (cosTmp - aTmp)            # size along track
        self.area = self.dS * self.dT
        # Overlap along the swath direction
        self.dist_pixels_S = dd[1:] - dd[:-1]   # minus one element
        self.overlap_S = ((self.dS[1:] + self.dS[:-1]) * 0.5 / self.dist_pixels_S - 1)  # fraction
        # overlap along track is based on 2063 km length of a single swatch (approx. value)
        self.overlap_T = (self.dT - (2063.0 / 2030)) / self.dT   # fraction
        self.area_corr = self.area / (1. - self.overlap_T)


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
        geo_1km = (np.zeros(shape=(lon_5km.shape[0]*5,lon_5km.shape[1]*5+4)) * np.nan,   # lon_1km
                   np.zeros(shape=(lon_5km.shape[0]*5,lon_5km.shape[1]*5+4)) * np.nan)   # lat_1km

        # both longitude and latitude are interpolated the same way
        # Initialise...
        geo_1km[0][2::5, 2:-4:5] = lon_5km[:,:]
        geo_1km[1][2::5, 2:-4:5] = lat_5km[:,:]
        a = np.arange(0.0, 1.0, 0.2)   # linear interpolation coefficients
        # Sequentially applying the interpolation
        for iG in [0,1]:
            for shift in range(1,5):
                geo_1km[iG][2+shift:-3:5, 2:-4:5] = (geo_1km[iG][2:-3:5, 2:-4:5] * (1.-a[shift]) + 
                                                     geo_1km[iG][7::5,   2:-4:5] * a[shift])
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
            for shift in range(1,5):
                lon_1km_tmp[2+shift:-3:5, 2:-4:5] = (lon_1km_tmp[2:-3:5, 2:-4:5] * (1.-a[shift]) + 
                                                     lon_1km_tmp[7::5,   2:-4:5] * a[shift])
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

        # inner box is OK but outer bands, 2 cells wide from 3 sides, 6 cell wide from swath end
        # need extrapolation
        # Note that at -180 <-> 180 switch extrapolation may be not possible
        for iGeo in [0,1]:
            ny5km = lon_5km.shape[1]
            # start of ix. Cut y-wise ends, which are nans and cause warnings
            if np.any(np.abs(geo_1km[iGeo][2,2:ny5km*5-2] - 
                             geo_1km[iGeo][3,2:ny5km*5-2]) > 200.):
#            if np.any(np.abs(geo_1km[iGeo][2,:] - geo_1km[iGeo][3,:]) > 200.):
                geo_1km[iGeo][2:4,:][geo_1km[iGeo][2:4,:] < 0.] += 360.
            for i in [1, 0]:  
                geo_1km[iGeo][i,:] = 2. * geo_1km[iGeo][i+1,:] - geo_1km[iGeo][i+2,:]
            # end of ix
            nx = geo_1km[iGeo].shape[0]
            if np.any(np.abs(geo_1km[iGeo][nx-3,2:ny5km*5-2] - 
                             geo_1km[iGeo][nx-4,2:ny5km*5-2]) > 200.):
#            if np.any(np.abs(geo_1km[iGeo][nx-3,:] - geo_1km[iGeo][nx-4,:]) > 200.):
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
        self.lon_1km = geo_1km[0]
        self.lon_1km[self.lon_1km > 180.] -= 360.
        self.lon_1km[self.lon_1km < -180.] += 360.
        self.lat_1km = geo_1km[1]

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
        granBits = unpacked_cloud_fields()
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
        arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templCloudFiles)))
        if len(arFNms) == 1:
            if ifDebug: self.log.log('Reading Cloud data from ' + path.split(arFNms[0])[1])
            fIn = nc4.Dataset(arFNms[0],'r')
        elif len(arFNms) > 1:
            self.log.log('Several cloud files satisfy template:' + str(arFNms))
            fIn = nc4.Dataset(arFNms[-1],'r')
#            return False
        else:
            self.log.log('No files for template:' + self.templCloudFiles)
            return False
        #
        # geolocation: 5 km and turn to 1km.
        #
        # first, check for masked values
        f1 = fIn.variables['Longitude'][:,:]
        f2 = fIn.variables['Latitude'][:,:]
        try: lon5km = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f1.data)
        except: lon5km = f1
        try: lat5km = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f2.data)
        except: lat5km = f2
        #
        # If grid is given, can check that the granule is inside and return if not
        if grid_to_fit_in is not None:
            fxSwath, fySwath = self.grid.geo_to_grid(lon5km, lat5km)
            ixSwath = np.round(fxSwath).astype(np.int)
            idxX_OK = np.logical_and(ixSwath >= 0, ixSwath < self.grid.nx)
            if not np.any(idxX_OK) : return False
            iySwath = np.round(fySwath).astype(np.int)
            idxY_OK = np.logical_and(iySwath >= 0, iySwath < self.grid.ny)
            if not np.any(idxY_OK) : return False
        # Something is inside the grid. Proceed
        self.MODIS_1km_geo_from_5km(lon5km, lat5km)
        #
        # Get the cloud mask
        #
#        bits = np.unpackbits(fIn.variables['Cloud_Mask'][:,:,:].data, axis=0)
        # direct way fails, use this way - checked to work
        cld_packed = fIn.variables['Cloud_Mask'][:,:,:].data
        cld_bytes = np.zeros(shape=cld_packed.shape, dtype=np.uint8)  #'uint8')
        cld_bytes[:,:,:] = cld_packed[:,:,:]
        bits = np.unpackbits(cld_bytes,axis=0)
        #
        # Unpack the first byte 
        #
        self.BitFields = self.unpack_MxD03_byte_1(bits[:8,:,:])

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
        arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
        if len(arFNms) == 1:
            if ifDebug: print('Reading FRP from ', path.split(arFNms[0])[1])
            fIn = nc4.Dataset(arFNms[0], 'r')
        elif len(arFNms) > 1:
            self.log.log('Several FRP files satisfy template:' + str(arFNms))
            fIn = nc4.Dataset(arFNms[-1], 'r')
#            return False
        else:
            print('Template: ', self.templFRPFiles)
            print('Found files: ', glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
            return False
        #
        # Get the basic information on the fires - same set as for IS4FIRES v.2.0
        #
        self.FP_frp = fIn.variables['FP_power'][:]
        if self.FP_frp.shape[0] > 0:
            # non-empty granule
            self.FP_iLine = fIn.variables['FP_line'][:]     # probably index along the track [0:2029]
            self.FP_iSample = fIn.variables['FP_sample'][:] # probably, along the swath, [0:1353]
            self.FP_lon = fIn.variables['FP_longitude'][:]
            self.FP_lat = fIn.variables['FP_latitude'][:]
#            print('min-max of lines along line and across sample',
#                  np.min(self.FP_iLine), np.max(self.FP_iLine), 
#                 np.min(self.FP_iSample), np.max(self.FP_iSample))
            self.FP_dS = self.dS[self.FP_iSample]   # depends on swath 
            self.FP_dT = self.dT[self.FP_iSample]   # also depends on swath
            self.FP_T4 = fIn.variables['FP_T21'][:]
            self.FP_T4b = fIn.variables['FP_MeanT21'][:]
            self.FP_T11 = fIn.variables['FP_T31'][:]
            self.FP_T11b = fIn.variables['FP_MeanT31'][:]
            self.FP_TA = fIn.variables['FP_MeanDT'][:]
            if ifDebug: self.log.log('Nbr of fires: %g, %s'  % (self.FP_frp.shape[0], str(self.now_UTC)))
#        else:
            # No fires. Zeroes or missing data or half-half?
            # e.g. sea means zeroes, clouds - missing, etc.
#            self.log.log('No fires')

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
        self.FP_frp = fIn.variables['FP_power'][:]
        if len(self.FP_frp) == 0: return True         # empty granule
        self.FP_iLine = fIn.variables['FP_line'][:]     # probably index along the track [0:2029]
        self.FP_iSample = fIn.variables['FP_sample'][:] # probably, along the swath, [0:1353]
        self.FP_lon = fIn.variables['FP_longitude'][:]
        self.FP_lat = fIn.variables['FP_latitude'][:]
        print(np.min(self.FP_iLine), np.max(self.FP_iLine), 
              np.min(self.FP_iSample), np.max(self.FP_iSample))
        self.FP_dS = self.dS[self.FP_iSample]
        self.FP_dT = self.dT[self.FP_iSample]
        self.FP_T4 = fIn.variables['FP_T21'][:]
        self.FP_T4b = fIn.variables['FP_MeanT21'][:]
        self.FP_T11 = fIn.variables['FP_T31'][:]
        self.FP_T11b = fIn.variables['FP_MeanT31'][:]
        self.FP_TA = fIn.variables['FP_MeanDT'][:]
        self.FP_SolZenAng = fIn.variables['FP_SolZenAng']
        
        return True


    #===================================================================
    
    def write_granule_IS4FIRES_v2_0(self, iStartFire, fOut):
        # write it down
        # fireNbr yr mon day hr min sec lon lat dS dT km frp MW T4 T4b T11 T11b TA MCE FireArea
        for iFire in range(self.FP_frp.shape[0]):
            fOut.write('fire = %03i %s %g %g %g %g km %g MW %g %g %g %g %g %g %g %g\n' %
                       (iFire + iStartFire, self.now_UTC.strftime('%Y %m %d %H %M 0.0'), 
                        self.FP_lon[iFire], self.FP_lat[iFire], self.FP_dS[iFire], self.FP_dT[iFire],
                        self.FP_frp[iFire], self.FP_T4[iFire], self.FP_T4b[iFire], self.FP_T11[iFire], 
                        self.FP_T11b[iFire], self.FP_TA[iFire], 0.5, 0.01,   # FP_MCE[iFire], FP_FireArea[iFire))
                        self.FP_SolZenAng[iFire]))
# SILAm reads like this:
#        strTmp, strTmp1, iFire, yr, mon, day, hr, mn, sec, &
#                       & fLonTmp, fLatTmp, &
#                       & fDx, fDy, chSizeUnit, fFRP, chFRPUnit, &
#                       & fpT4, fpT4b, fpT11, fPT11b, fpTA, fpMCE, fParea
        return iStartFire + self.FP_frp.shape[0]


    #===================================================================

    def draw_granule(self, chOutFNm):
        #
        # Area covered by this swath, with a bit free space around:

        fig, axes = mpl.pyplot.subplots(4,2, figsize=(10,16))
        minLon = np.nanmin(self.lon_1km)
        minLat = np.nanmin(self.lat_1km)
        maxLon = np.nanmax(self.lon_1km)
        maxLat = np.nanmax(self.lat_1km)
        cmap = mpl.pyplot.get_cmap('cool')
        ixAx = 0
        iyAx = 0
        chFires = ' no fires'
        for var, chTxt, cm in [(self.BitFields.ifCloudAnalysed,'Cloud mask(1=made) + FRP, [MW]','cool'),
                               (self.BitFields.QA,'QA: 0=cld,1=?,10=clr?,11=clr','Paired'),
                               (self.BitFields.day_night, '1=day, 0=night','cool'), 
                               (self.BitFields.sunglint,'0=sunglint','cool'),
                               (self.BitFields.snow,'0=snow','cool'),
                               (self.BitFields.land,'0=water,1=coast,10=desert,11=land','Paired'),
                               (self.lon_1km,'Lon','cool'),
                               (self.lat_1km,'Lat','cool')]:
            ax = axes[ixAx,iyAx]
            print('Plotting ', chTxt)
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
            cs = bmap.scatter(self.lon_1km[::2,::2].T, self.lat_1km[::2,::2].T, c=var[::2,::2].T, 
                              s=1, edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap(cm))
            cbar = bmap.colorbar(cs,location='bottom',pad="7%")
#            cbar.set_label(chTxt, fontsize=9)
            ax.set_title(chTxt, fontsize=10)
            if ixAx + iyAx == 0 and self.FP_frp.shape[0] > 0:
                print('Plotting FRP')
                chFires = ', %g fires' % self.FP_frp.shape[0]
                sort_order = self.FP_frp.argsort()
                cs2 = bmap.scatter(self.FP_lon[sort_order], self.FP_lat[sort_order], 
                                   c=self.FP_frp[sort_order], s=30, 
                                   edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap('rainbow'))
                cbar2 = bmap.colorbar(cs2,location='right',pad="7%")
                cbar2.set_label('FRP, MW', fontsize=9)
            if ixAx == 3:
                ixAx = 0
                iyAx += 1
            else: ixAx += 1

        # If FRP added, remove labels on the right-hand-side map
        if self.FP_frp.shape[0] > 0:
            axes[0,1].yaxis.set_ticklabels([])
        mpl.pyplot.suptitle(path.split(chOutFNm)[1].strip('.png') + chFires, fontsize=14)
        mpl.pyplot.savefig(chOutFNm,dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()

