'''
Created on 5.3.2023

 The class holding the information on a single swatch of the SLSTR satellite instrument 

Functions available:
- __init__:   Sets basic file names 
- get_pixel_size:  basic SLSTR geometry

@author: sofievm
'''
# A couple of constants

import numpy as np
import numpy.f2py
import scipy.interpolate as interp
import glob, datetime as dt, shutil
import os, sys
import matplotlib as mpl
import netCDF4 as nc4
from mpl_toolkits import basemap
import granule__basic as gb
from toolbox import drawer, supplementary as spp, silamfile, gridtools


ifDebug = True


#=============================================================
def productType(chFNm):
    dicKnownProducts = {#'MOD03':('MODIS','auxiliary'), 'MYD03':('MODIS','auxiliary'),
                        #'MOD14':('MODIS','fire'),      'MYD14':('MODIS','fire')
                        }
    for satProd in dicKnownProducts.keys():
        if satProd in chFNm: return dicKnownProducts[satProd]
    return 'unknown'



#################################################################################
#
# The class holding the information on a single swatch of the satellite, 
#
#################################################################################

class granule_SLSTR(gb.granule_basic):

    #=============================================================
    def __init__(self, now_UTC=dt.datetime.utcnow(), 
                 chCaseName='', granType='', log=None):
        #
        # Initialises the basic object accounting for SLSTR specifics, i.e., 4 grids
        # Radiances are stored in 0.5km _an-grid (channels S1-S6) and _bn-grid (S4-S6, b-stripe)
        # 1km i-grid (IR channels S7-S9 and F2), and 1km f-grid (IR channel F1)
        # FRP is taken from i- and f-grids, they are the ones available from FRP products,
        # so they are the ones handled here. 
        # There are 2 channels with different grids: _fn for 1km dedicated F1-channel and
        # _in for 1km grid of other IR channels. There is also tie-point _tn, whose grid is not
        # given in the geo files - it is used for parameters where 16km resolution is enough.
        #
        # All this mess is handled by SLSTR repacking procedure, we do not need it.
        # Here, we allow only NRT (Near-Real-Time) SLSTR original or repacked datasets.
        # The reason is: NRT need to be handled in operational routine, so it makes sense to 
        # process it in one gulp. Otherwise, one first has to repack the dataset and only then use it. 
        # 
        self.chCaseNm = chCaseName
        self.granType = granType
        self.type = 'SLSTR'
        if granType == '':  # placeholder
            gb.granule_basic.__init__(self, self.type, now_UTC, '', '', log, (log is None) or (not ifDebug))
        elif granType == 'NRT':  # NRT original format
            gb.granule_basic.__init__(self, self.type, now_UTC, 'frp_in', ['geodetic_in','flags_in'], 
                                      log, (log is None) or (not ifDebug))
        elif granType == 'repacked':  #our condensed format
            gb.granule_basic.__init__(self, self.type, now_UTC, 'SLSTR_FRP', 'SLSTR_GEO', 
                                      log, (log is None) or (not ifDebug))
            
        self.get_pixel_size()        # get the distribution of swath

    #=============================================================
    def get_FRP_FNm(self, chGridID=''):
        if self.granType == '': return None
        elif self.granType == 'NRT': return os.path.join(self.chCaseNm.strftime(self.now_UTC),'frp_in.nc')
        elif self.granType == 'repacked': return self.chCaseNm.strftime(self.now_UTC) + '_FRP.nc4'
        else: raise ValueError('Wrong granule type (must be <empty>, NRT or repacked):' + self.granType)

    
    #=============================================================
    def get_geo_FNm(self, chGridID=''):
        if self.granType == '': return None
        elif self.granType == 'NRT': return os.path.join(self.chCaseNm.strftime(self.now_UTC),'geodetic_in.nc')
        elif self.granType == 'repacked': return self.chCaseNm.strftime(self.now_UTC) + '_GEO.nc4'
        else: raise ValueError('Wrong granule type (must be <empty>, NRT or repacked):' + self.granType)

 
    #=============================================================
    def abbrev_1(self, chFNm):
        return 'S'.encode('utf-8')


    #=============================================================

    def get_pixel_size(self):
        #
        # An approximate but quite accurate formula for dS and dT pixel size along the scan 
        # and along the track directions, respectively
        #
        # General formula is presented in Ichoku & Kaufman, IEEE Transactions 2005
        #
        # SLSTR parameters are taken from:
        # https://sentinels.copernicus.eu/documents/247904/4598082/Sentinel-3-SLSTR-Land-Handbook.pdf
        #
        # Wooster, M., Xu, W., and Nightingale, T.: Sentinel-3 SLSTR active ﬁre detection and FRP product: 
        # Pre-launch algorithm development and performance evaluation using MODIS and ASTER datasets, 
        # Remote Sensing of Environment, 120, 236–254, https://doi.org/10.1016/j.rse.2011.09.033, 2012.
        #
        # Coppo, P., Ricciarelli, B., Brandani, F., Delderfield, J., Ferlet, M., Mutlow, C., et al.(2010). 
        # SLSTR: A high accuracy dual scan temperature radiometer for sea and land surface monitoring
        # from space. Journal of Modern Optics, 57(18), 1815–1830 October 2010.
        #
        self.h = 814.5      # SLSTR altitude, km.
#        self.oblique_view = 55  # degrees, zenith angle, dual-view, not used for FRP
        self.view_width = 1407  # km, swatch width
         
        self.N = 1500     # pixels along scan
        self.N_track = 2000
        self.minAngle = np.float64(-47 * np.pi / 180.)
        self.maxAngle = np.float64(30 * np.pi / 180.)     # Asymmetric!
        self.scanSz = 1   # how many scan lines in one scan
        r = gb.R_Earth + self.h
        # scan geometry
        sz_scan_nadir = 1.0   # km Coppo ea, Nominal size of nadir pixel along scan. Actually, triangle (Fig.8)
        self.theta =  np.array(list( (self.minAngle + i*(self.maxAngle-self.minAngle)/(self.N-1)
                                         for i in range(self.N))))
        aTmp = np.sqrt((gb.R_Earth/ r)**2 - np.square(np.sin(self.theta)))
        self.dS = gb.R_Earth * sz_scan_nadir / self.h * ((np.cos(self.theta)/ aTmp) - 1.0)   # size along scan, no aggregation
        # pixel size along track
        self.sz_track_nadir = 2.0 # km Coppo et al, size of nadir pixel along track
        self.dT = (r * self.sz_track_nadir / self.h *             # size along track
                   (np.cos(self.theta) - np.sqrt((gb.R_Earth/ r)**2 - np.square(np.sin(self.theta)))))

        # distance from swatch centre point
        c = r * np.cos(self.theta) - np.sqrt( r*r* np.square(np.cos(self.theta)) + gb.R_Earth*gb.R_Earth - r*r)
        self.dist_nadir = gb.R_Earth * np.arcsin(c * np.sin(self.theta) / gb.R_Earth)
        self.area = self.dS * self.dT
        # Overlap along the scan direction (bow-tie)
        self.dist_pixels_S = self.dist_nadir[1:] - self.dist_nadir[:-1]   # minus one element
        self.overlap_S = ((self.dS[1:] + self.dS[:-1]) * 0.5 / self.dist_pixels_S - 1)  # fraction
        # overlap along track is based on 2063 km length of a single swath (approx. value)
        self.overlap_T = self.dT / np.min(self.dT) - 1  # fraction
        self.area_corr = self.area / (1. - self.overlap_T)


    #=======================================================================================
    
    def restore_geo_fields(self, chCaseNm):
        #
        # SLSTR uses very bad arrangement of the L2 files. The lon-lat granule has to be restored from
        # several files.
        #
        chFNmGeodetic = 'd:\\data\\satellites\\SLSTR\\NTC_not_time_critical\\S3A_SL_2_FRP____20230304T065159_20230304T065459_20230305T153712_0179_096_134_3780_PS1_O_NT_004.SEN3\\geodetic_in.nc'
        chFNmIndices = 'd:\\data\\satellites\\SLSTR\\NTC_not_time_critical\\S3A_SL_2_FRP____20230304T065159_20230304T065459_20230305T153712_0179_096_134_3780_PS1_O_NT_004.SEN3\\indices_in.nc'
        
        print('File selection does not work yet')
        
        self.lon, self.lat, self.height = self.process_geodetic(chFNmGeodetic, chFNmIndices)
        return 


    #=======================================================================================

    def process_geodetic(self, chFNmGeodetic, chFNmIndices):
        #
        # Receives two file names - the geodetic file and the corresponding index file, merges them
        # and obtains lon and lat fields
        #
        fInGeo = nc4.Dataset(chFNmGeodetic, "r")
        fInGeo.set_auto_maskandscale(False)
        fInIdx = nc4.Dataset(chFNmIndices, 'r')
        fInIdx.set_auto_maskandscale(False)
        #
        # Variables are structures, combined into a dictionary: variables['latitude_fn'].units
        #
        # geography
        chLon=None; chLonOrph=None; chLat=None; chLatOrph=None
        for var in fInGeo.variables.keys():
            chTypeGeo = var.split('_')[-1]
            if var.startswith('longitude_'):
                if 'orphan' in var: chLonOrph = var
                else: chLon = var
            if var.startswith('latitude_'):
                if 'orphan' in var: chLatOrph = var
                else: chLat = var
            if var.startswith('elevation_'):
                if 'orphan' in var: chHeightOrph = var
                else: chHeight = var
        try: chLon + chLonOrph + chLat + chLatOrph + chHeight + chHeightOrph
        except: print(chLon, chLonOrph, chLat, chLatOrph, chHeight, chHeightOrph)
        # Indices
        chPix=None; chPixOrph=None; chScan=None; chScanOrph=None
        for var in fInIdx.variables.keys():
            chTypeIdx = var.split('_')[-1]
            if var.startswith('pixel_'):
                if 'orphan' in var: chPixOrph = var
                else: chPix = var
            if var.startswith('scan_'):
                if 'orphan' in var: chScanOrph = var
                else: chScan = var
            if chTypeGeo != chTypeIdx:
                raise ValueError('Geo and Idx types do not agree: ' + chTypeGeo + ', ' + chTypeIdx)
        if not (chPix, chPixOrph, chScan, chScanOrph):
            print(chPix, chPixOrph, chScan, chScanOrph)

        print(chLon, fInGeo.variables[chLon].shape, chLonOrph, fInGeo.variables[chLonOrph].shape)
        print(chLat, fInGeo.variables[chLat].shape, chLatOrph, fInGeo.variables[chLatOrph].shape)
        print(chPix, fInIdx.variables[chPix].shape, chPixOrph, fInIdx.variables[chPixOrph].shape)
        print(chScan,fInIdx.variables[chScan].shape,chScanOrph,fInIdx.variables[chScanOrph].shape)
        # Missing values and scaling
        arLon = fInGeo.variables[chLon][:,:].astype(np.float32) * fInGeo.variables[chLon].scale_factor
        arLonOrphan = fInGeo.variables[chLonOrph][:,:].astype(np.float32) * fInGeo.variables[chLonOrph].scale_factor
        arLat = fInGeo.variables[chLat][:,:].astype(np.float32) * fInGeo.variables[chLat].scale_factor
        arLatOrphan = fInGeo.variables[chLatOrph][:,:].astype(np.float32) * fInGeo.variables[chLatOrph].scale_factor
        arHeight = fInGeo.variables[chHeight][:,:].astype(np.float32) * fInGeo.variables[chHeight].scale_factor
        arHeightOrphan = fInGeo.variables[chHeightOrph][:,:].astype(np.float32) * fInGeo.variables[chHeightOrph].scale_factor
        arLon[abs(arLon-fInGeo.variables[chLon]._FillValue) < 
              1e-5 * abs(arLon+fInGeo.variables[chLon]._FillValue)] = np.nan
        arLonOrphan[abs(arLonOrphan-fInGeo.variables[chLonOrph]._FillValue) <
                    1e-5 * abs(arLonOrphan+fInGeo.variables[chLonOrph]._FillValue)] = np.nan
        arLat[abs(arLat-fInGeo.variables[chLat]._FillValue) <
              1e-5 * abs(arLat+fInGeo.variables[chLat]._FillValue)] = np.nan
        arLatOrphan[abs(arLatOrphan-fInGeo.variables[chLatOrph]._FillValue) <
                    1e-5 * abs(arLatOrphan+fInGeo.variables[chLatOrph]._FillValue)] = np.nan
        arHeight[abs(arHeight-fInGeo.variables[chHeight]._FillValue) < 
              1e-5 * abs(arHeight+fInGeo.variables[chHeight]._FillValue)] = np.nan
        arHeightOrphan[abs(arHeightOrphan-fInGeo.variables[chHeightOrph]._FillValue) <
                    1e-5 * abs(arHeightOrphan+fInGeo.variables[chHeightOrph]._FillValue)] = np.nan
        # Indices
        arPixIdx = fInIdx.variables[chPix][:,:]
        arPixIdxOrphan = fInIdx.variables[chPixOrph][:,:]
        arScanIdx = fInIdx.variables[chScan][:,:]
        arScanIdxOrphan = fInIdx.variables[chScanOrph][:,:]
        print(fInIdx.variables[chScan])
        print('PixIdx==fill, PixIdx OK:',
              np.nansum(arPixIdx==fInIdx.variables[chPix]._FillValue), np.sum(arPixIdx!=fInIdx.variables[chPix]._FillValue),
              np.nanmax(arPixIdx[arPixIdx!=fInIdx.variables[chPix]._FillValue]),
              np.nanmin(arPixIdx[arPixIdx!=fInIdx.variables[chPix]._FillValue]))
        print('PixIdxOrphan==fill, PixIdxOrphan OK:',
              np.nansum(arPixIdxOrphan==fInIdx.variables[chPixOrph]._FillValue), np.sum(arPixIdxOrphan!=fInIdx.variables[chPixOrph]._FillValue),
              np.nanmax(arPixIdxOrphan[arPixIdxOrphan!=fInIdx.variables[chPixOrph]._FillValue]),
              np.nanmin(arPixIdxOrphan[arPixIdxOrphan!=fInIdx.variables[chPixOrph]._FillValue]))
        print('ScanIdx==fill, ScanIdx OK:',
              np.nansum(arScanIdx==fInIdx.variables[chScan]._FillValue), np.sum(arScanIdx!=fInIdx.variables[chScan]._FillValue),
              np.nanmax(arScanIdx[arScanIdx!=fInIdx.variables[chScan]._FillValue]),
              np.nanmin(arScanIdx[arScanIdx!=fInIdx.variables[chScan]._FillValue]))
        print('ScanIdxOrphan==fill, ScanIdxOrphan OK:',
              np.nansum(arScanIdxOrphan==fInIdx.variables[chScanOrph]._FillValue), np.sum(arScanIdxOrphan!=fInIdx.variables[chScanOrph]._FillValue),
              np.nanmax(arScanIdxOrphan[arScanIdxOrphan!=fInIdx.variables[chScanOrph]._FillValue]),
              np.nanmin(arScanIdxOrphan[arScanIdxOrphan!=fInIdx.variables[chScanOrph]._FillValue]))
        #
        # Fill-in the coordinate arrays
        #
        arLonFinal = arLon.copy() * np.nan
        arLatFinal = arLat.copy() * np.nan
        arHeightFinal = arHeight.copy() * np.nan
        minScan = min(np.nanmin(arScanIdx), np.nanmin(arScanIdxOrphan))
        # The main array
        iPixMiss = fInIdx.variables[chPix]._FillValue
        iScanMiss = fInIdx.variables[chScan]._FillValue
        for iPix in range(arPixIdx.shape[0]):
            for iScan in range(arScanIdx.shape[1]):
                if arPixIdx[iPix,iScan] == iPixMiss:
                    if arScanIdx[iPix,iScan] == iScanMiss: continue
                    else: raise ValueError('pix is missing, scan is not')
                if arScanIdx[iPix,iScan] == iScanMiss: raise ValueError('scan is missing, pix is not')
                arLonFinal[arPixIdx[iPix,iScan], arScanIdx[iPix,iScan] - minScan] = arLon[iPix,iScan]
                arLatFinal[arPixIdx[iPix,iScan], arScanIdx[iPix,iScan] - minScan] = arLat[iPix,iScan]
                arHeightFinal[arPixIdx[iPix,iScan], arScanIdx[iPix,iScan] - minScan] = arHeight[iPix,iScan]
        # Orphan patches
        iPixMiss = fInIdx.variables[chPixOrph]._FillValue
        iScanMiss = fInIdx.variables[chScanOrph]._FillValue
        for iPix in range(arPixIdxOrphan.shape[0]):
            for iScan in range(arScanIdxOrphan.shape[1]):
                if arPixIdxOrphan[iPix,iScan] == iPixMiss:
                    if arScanIdxOrphan[iPix,iScan] == iScanMiss: continue
                    else: raise ValueError('orphan pix is missing, scan is not')
                if arScanIdxOrphan[iPix,iScan] == iScanMiss: raise ValueError('orphan scan is missing, pix is not')
                arLonFinal[arPixIdxOrphan[iPix,iScan], arScanIdxOrphan[iPix,iScan] - minScan] = arLonOrphan[iPix,iScan]
                arLatFinal[arPixIdxOrphan[iPix,iScan], arScanIdxOrphan[iPix,iScan] - minScan] = arLatOrphan[iPix,iScan]
                arHeightFinal[arPixIdxOrphan[iPix,iScan], arScanIdxOrphan[iPix,iScan] - minScan] = arHeightOrphan[iPix,iScan]
        
        return (arLonFinal, arLatFinal, arHeightFinal)



    #===================================================================================
    
    def py_upsample_fld (self, fIn, scanSz_In_tr, pixel_nbr_scaling_along_track, pixel_nbr_scaling_along_scan, nPixT_out, nPixS_out, nPixT_in, nPixS_in):
        #
        # Reverse to downsample: inter-/extra-polates the low-resolution field to high-resolving one
        # Procedure is scan-wise and includes two-steps: (i) each low-res line along-track-within-scan is
        # filled with linear inter-extrapolation, (ii) these lines are inter-extrapolated along the scan
        #
        # nullify output map
        fOut = np.zeros(shape=(nPixT_out, nPixS_out))
        line = np.zeros(shape=(scanSz_In_tr*pixel_nbr_scaling_along_track, nPixS_in))
  
        scanSz_Out_tr = scanSz_In_tr * pixel_nbr_scaling_along_track  # scan size along track of the output

        # Positions of low-res-points within the high-res scan
        pos_tr = np.array(list((pixel_nbr_scaling_along_track / 2.- 0.5 + pixel_nbr_scaling_along_track * iTmp 
                                for iTmp in range(scanSz_In_tr))), dtype=np.float32)  # along-track
        pos_sc = np.array(list((pixel_nbr_scaling_along_scan / 2.- 0.5 + pixel_nbr_scaling_along_scan * iTmp 
                                for iTmp in range(nPixS_in))), dtype=np.float32)  # along-scan
  
        if not self.ifSilent: 
            print ("scanSz_In_tr, nPixS_in", scanSz_In_tr, nPixS_in)
            print ("pos_tr ", pos_tr)
            print ("pos_sc: ", pos_sc)
        
        # Proceed scan-by-scan
        #
        for iSc in range(round(nPixT_in / scanSz_In_tr)):   # number of scans
            #
            # Fill-in the along-track lines. Note that they are shifted by 0.5 in both directions
            # Interpolation goes from two reference points. When reaching the further one, shift
            #
            iLR_0 = 0  # the first low-res reference point
            iLR_1 = 1  # the second low-res reference point
    
            slope = (fIn[iSc * scanSz_In_tr + iLR_1, :] -
                     fIn[iSc * scanSz_In_tr + iLR_0, :]) / (pos_tr[iLR_1] - pos_tr[iLR_0])

            for iTmp in range(scanSz_Out_tr):          # cycle over high-res line along track
                if iTmp >= pos_tr[iLR_1]:             # reached the farther ref point?
                    iLR_1 = np.minimum(iLR_1+1, scanSz_In_tr-1)    # allow extrapolation at the end
                    iLR_0 = iLR_1 - 1
                line[iTmp,:] = fIn[iSc * scanSz_In_tr + iLR_0, :] + slope[:] * (iTmp - pos_tr[iLR_0])
    
#        for iSc in range(round(nPixT_in / scanSz_In_tr)):   # number of scans
            #
            # Step 2: intermeditate lines to the whole scan. The procedure is transpose to the above
            #
            iLR_0 = 0  # the first low-res reference point
            iLR_1 = 1  # the second low-res reference point
            for iTmp in range(nPixS_out):          # cycle over high-res scan
                if iTmp >= pos_sc[iLR_1]:          # reached the further ref point?
                    iLR_1 = np.minimum(iLR_1+1, nPixS_in-1)     # allow extrapolation at the end
                    iLR_0 = iLR_1 - 1
                fOut[iSc * scanSz_Out_tr : (iSc + 1) * scanSz_Out_tr, iTmp
                     ] = line[:,iLR_0] + (line[:,iLR_1] - 
                                          line[:,iLR_0]) * (iTmp - pos_sc[iLR_0]) / (pos_sc[iLR_1] - pos_sc[iLR_0])
        return fOut


    #=============================================================

    def downsample_field(self, DS_factor_track, DS_factor_scan, fldIn, ifLongitude):
        #
        # Downsamples the granule: instead of 32 pixels along the track in a single scan, 
        # retains a subset as defined in the DS_table. In each type of downsampling,
        # a square 2x2 of IMG pixels is used to calculate the point to store. 
        # Note that aggregation is performed in good places: 1280 and 2016, which allow
        # up to 32-fold downsampling. Note that with 32 lines per scan max downscaling
        # along the track is 16.
        #
        if DS_factor_track == 1 and DS_factor_scan == 1: return fldIn
        if not (DS_factor_track, DS_factor_scan) in [(2,2),(4,4),(8,8),(16,16),(16,32)]:
            raise ValueError('Downsample factor must be a power of 2')
        #
        # Dimension 1 is along the scan
        #
        fResolScale = self.N_375 // fldIn.shape[1]
        idx1_tr = np.array(range(DS_factor_scan // 2 - 1, self.N_375 // fResolScale, DS_factor_scan), dtype=np.int16)
        idx1_sc = np.array(range(DS_factor_track // 2 - 1, self.scanSz_375 // fResolScale, DS_factor_track), dtype=np.int16)
        # Scans are processed individually, so reshaping is nice. Should not copy the data
        fIn = fldIn.reshape((-1,self.scanSz_375,self.N_375))  # (202,32,6400)
        # New field (nScans, pixels_in_scan_along_track, pixels_along_scan)
        f_av = np.zeros(shape=(fIn.shape[0], idx1_sc.size, idx1_tr.size), dtype=fldIn.dtype)
        # Generic step
        f_av[:,:,:] = (fIn[:,idx1_sc][:,:,idx1_tr] + 
                       fIn[:,idx1_sc][:,:,idx1_tr+1] +
                       fIn[:,idx1_sc+1][:,:,idx1_tr] + 
                       fIn[:,idx1_sc+1][:,:,idx1_tr+1]) / 4.0
#        # construct the spline: may be, smaller or more accurate?
#          NOT REALLY: much bigger. May be, in the up-sampling...
#        bspl = interp.make_interp_spline(range(self.N_375),fIn[0,0,:], k=3, t=None, bc_type='natural', axis=0, check_finite=True)
        #
        # If the downsampling is done in lonlat grid and this is longitude, handle jump over
        # -180:180 line
        #
        if ifLongitude: self.rotate_longitude(idx1_sc, idx1_tr, fIn, f_av, fldIn.dtype)
        
        return f_av.reshape(f_av.shape[0] * f_av.shape[1], f_av.shape[2])


    #=============================================================
    
    def downsample_granule_geo(self, DS_factor_track, DS_factor_scan, ifReturnLonLat, chFNmOut=None):
        #
        # Downsamples the gelocation fields of a granule. Should the output file be given,
        # writes it down
        # The downsampled granule does not need adjustment with height, we just store the
        # interpolated one
        #
        self.DS_factor_track = DS_factor_track
        self.DS_factor_scan = DS_factor_scan
        # Convert to Cartesian
        x, y, z = gridtools.xyzFromLonLat(self.lon, self.lat)
        # downsample in Cartesian
        self.height_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, self.height, False)
        # return back to lonlat
        if ifReturnLonLat:
            x_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, x, False)
            y_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, y, False)
            z_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, z, False)
            self.lon_DS, self.lat_DS = gridtools.lonlatFromXYZ(x_DS, y_DS, z_DS)
            self.granule_type = 'downsampled_lonlat'
        else:
            self.x_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, x, False)
            self.y_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, y, False)
            self.z_DS = self.downsample_field(self.DS_factor_track, self.DS_factor_scan, z, False)
            self.granule_type = 'downsampled_cartesian'
        # if output name is given, store the granule
        if not chFNmOut is None:
            self.geolocation_2_nc(True, chFNmOut)   # ifDownsampled


    #=================================================================
    
    def restore_granule_geo(self):
        #
        # Restores the granule geolocation fields to the nominal 375 m resolution
        #
        # Convert to Cartesian
        if self.granule_type == 'downsampled_lonlat':
            x_DS, y_DS, z_DS = gridtools.xyzFromLonLat(self.lon_DS, self.lat_DS)
        elif self.granule_type == 'downsampled_cartesian':
            x_DS = self.x_DS
            y_DS = self.y_DS
            z_DS = self.z_DS
        else:
            raise ValueError('Unknown granule type:' + self.granule_type)
        #
        # Restore all three coordinates
        #
        x = f_granule_viirs.f_upsample_fld(x_DS,                             # input field 
                                           round(self.scanSz_375 / self.DS_factor_track), # scanSz_In_tr
                                           self.idx_segments,            # edges of 375m segments
                                           self.DS_factor_track, self.DS_factor_scan,
                                           x_DS.shape[0] * self.DS_factor_track, self.N_375,  # nPixT_out, nPixS_out,  6400     # total number of I-band pixels along the scan
                                           x_DS.shape[0], x_DS.shape[1]) # nPixT_in, nPixS_in)
        y = f_granule_viirs.f_upsample_fld(y_DS,                             # input field 
                                           round(self.scanSz_375 / self.DS_factor_track), # scanSz_In_tr
                                           self.idx_segments,            # edges of 375m segments
                                           self.DS_factor_track, self.DS_factor_scan, 
                                           y_DS.shape[0] * self.DS_factor_track, self.N_375,  # nPixT_out, nPixS_out,  6400     # total number of I-band pixels along the scan
                                           y_DS.shape[0], y_DS.shape[1]) # nPixT_in, nPixS_in)
        z = f_granule_viirs.f_upsample_fld(z_DS,                             # input field 
                                           round(self.scanSz_375 / self.DS_factor_track), # scanSz_In_tr
                                           self.idx_segments,            # edges of 375m segments
                                           self.DS_factor_track, self.DS_factor_scan, 
                                           z_DS.shape[0] * self.DS_factor_track, self.N_375,  # nPixT_out, nPixS_out,  6400     # total number of I-band pixels along the scan
                                           z_DS.shape[0], z_DS.shape[1]) # nPixT_in, nPixS_in)
        # Back to lonlat
        #
        self.lon, self.lat = gridtools.lonlatFromXYZ(x, y, z)
        #
        # Deal with the height correction
        # Upscale height
        #
        self.height = f_granule_viirs.f_upsample_fld(self.height_DS,                             # input field 
                                                     round(self.scanSz_375 / self.DS_factor_track), # scanSz_In_tr
                                                     self.idx_segments,            # edges of 375m segments
                                                     self.DS_factor_track, self.DS_factor_scan, 
                                                     self.height_DS.shape[0] * self.DS_factor_track, 
                                                     self.N_375,  # nPixT_out, nPixS_out,  6400     # total number of I-band pixels along the scan
                                                     self.height_DS.shape[0], # nPixT_in
                                                     self.height_DS.shape[1]) # nPixS_in)
        #
        # Now, get the global height file and adjust the restored geodata
        # Note that the file is large, reader is too heavy. Do it yourself
        #
        if not self.glob_height_metadata:
            self.glob_height_fIn = nc4.Dataset(self.chFNm_GlobHeight, 'r')
            self.glob_height_metadata = silamfile.SilamNCFile(self.chFNm_GlobHeight)
            try:
                self.glob_height = self.glob_height_fIn.variables['relief_height'][:,:].data
            except:
                self.glob_height = self.glob_height_fIn.variables['relief_height'][:,:]

        # let's return the true granule height
        return self.update_Cartesian_to_height(x, y, z, self.glob_height,self.glob_height_metadata.grid)
        


    #=============================================================

    def pick_granule_data_IS4FIRES_v3_0(self, grid_to_fit_in=None):
        #
        # SLSTR has a single zip with a bunch of files, including FRP and geolocation
        # Read them in.
        #
        # Start from geolocation
        #
        arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templAuxilFiles)))
        if len(arFNms) == 1:
            if not self.ifSilent: 
                self.log.log('Reading geolocation from ' + os.path.split(arFNms[0])[1])
            fIn = nc4.Dataset(arFNms[0],'r')
            self.chFNm_geo = arFNms[0]
        elif len(arFNms) > 1:
            if not self.ifSilent: 
                self.log.log('Several geolocation files satisfy template:' + str(arFNms))
            fIn = nc4.Dataset(arFNms[-1],'r')
#            return False
        else:
            if not self.ifSilent: 
                self.log.log(self.now_UTC.strftime('%Y%m%d_%H%M: No auxiliary files for template:') + 
                             self.templAuxilFiles)
            return False
        #
        # geolocation: what dataset do we have? 375m, 750m, or downsampled? Cartesian or lonlat?
        #
        try:
            self.granule_type = fIn.getncattr('file_type')
        except:
            self.granule_type = 'external'
        # process appropriately
        if self.granule_type == 'downsampled_lonlat':
            # downsampled lonlat
            self.DS_factor_track = fIn.getncattr('downsampling_factor_along_track')    # available only for downsampled set
            self.DS_factor_scan = fIn.getncattr('downsampling_factor_along_scan')    # available only for downsampled set
            self.lon_DS = fIn['longitude'][:,:]
            self.lat_DS = fIn['latitude'][:,:]
            self.height_DS = fIn['height'][:,:]
            self.restore_granule_geo_Cartesian(True)             # Make the full-resolution lon-lat
        elif self.granule_type == 'downsampled_cartesian':
            # downsampled Cartezian
            self.DS_factor_track = fIn.getncattr('downsampling_factor_along_track')    # available only for downsampled set
            self.DS_factor_scan = fIn.getncattr('downsampling_factor_along_scan')    # available only for downsampled set
            self.x_DS = fIn['x'][:,:]
            self.y_DS = fIn['y'][:,:]
            self.z_DS = fIn['z'][:,:]
            self.height_DS = fIn['height'][:,:]
            self.restore_granule_geo_Cartesian(False)        # Make the full-resolution lon-lat
        elif self.granule_type == 'external':
            # Full- or half- resolution lon-lat
            fInInt = fIn['HDFEOS']['SWATHS']
            try:
                # full resolution
                f1 = fInInt['VNP_375M_GEOLOCATION']['Geolocation Fields']['Longitude'][:,:]
                f2 = fInInt['VNP_375M_GEOLOCATION']['Geolocation Fields']['Latitude'][:,:]
                f3 = fInInt['VNP_375M_GEOLOCATION']['Data Fields']['Height'][:,:]
                # check for masked values
                try: self.lon = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f1.data)
                except: self.lon = f1
                try: self.lat = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f2.data)
                except: self.lat = f2
                try: self.height = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f3.data)
                except: self.height = f3
            except:
                # Half resolution
                f1 = fInInt['VNP_750M_GEOLOCATION']['Geolocation Fields']['Longitude'][:,:]
                f2 = fInInt['VNP_750M_GEOLOCATION']['Geolocation Fields']['Latitude'][:,:]
                f3 = fInInt['VNP_750M_GEOLOCATION']['Data Fields']['Height'][:,:]
                # check for masked values
                try: self.lon_DS = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f1.data)
                except: self.lon_DS = f1
                try: self.lat_DS = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f2.data)
                except: self.lat_DS = f2
                try: self.height_DS = np.where(np.logical_or(f1.mask, f2.mask), np.nan, f3.data)
                except: self.height_SD = f3
                # Return the 375 fields
                self.DS_factor_track = 2
                self.DS_factor_scan = 2
                self.restore_granule_geo_Cartesian(True)
        #
        # Whatever file type is, attributes follow the same style
        # 
        # global attributes
        self.dicGlobal_attr = {}
        for a in fIn.ncattrs():
            self.dicGlobal_attr[a] = fIn.getncattr(a)
        # groups and group attributes
        self.dicGroups = {}
        for g in fIn.groups.keys():
            self.dicGroups[g] = {}
            for a in fIn[g].ncattrs():
                self.dicGroups[g][a] = fIn[g].getncattr(a)
        fIn.close()
        #
        # If grid is given, can check that the granule is inside and return if not
        #
        if grid_to_fit_in is not None:
            fxSwath, fySwath = self.grid.geo_to_grid(self.lon, self.lat)
            ixSwath = np.round(fxSwath).astype(np.int)
            idxX_OK = np.logical_and(ixSwath >= 0, ixSwath < self.grid.nx)
            if not np.any(idxX_OK) : return False
            iySwath = np.round(fySwath).astype(np.int)
            idxY_OK = np.logical_and(iySwath >= 0, iySwath < self.grid.ny)
            if not np.any(idxY_OK) : return False
        #
        # Clouds done, geolocation is known. Get FRP
        # FRP file has two parts: Fire-Pixel Table and packed maps - another set of bytes
        #
        arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
        if len(arFNms) == 1:
            if not self.ifSilent: 
                print('Reading FRP from ', os.path.split(arFNms[0])[1])
            fIn = nc4.Dataset(arFNms[0], 'r')
            self.chFNm_FRP = arFNms[0]
        elif len(arFNms) > 1:
            if not self.ifSilent: 
                self.log.log('Several FRP files satisfy template:' + str(arFNms))
            fIn = nc4.Dataset(arFNms[-1], 'r')
#            return False
        else:
            if not self.ifSilent: 
                self.log.log('Template: ' + self.templFRPFiles)
                self.log.log('Found files: ' + '\n'.join(glob.glob(self.now_UTC.strftime(self.templFRPFiles))))
                self.log.log('Something went wrong: no files for template')
            return False
        #
        # Get the cloud mask from v??_14
        #
        cld_packed = fIn['fire mask'][:,:].data
#        cld_bytes = np.zeros(shape=cld_packed.shape, dtype=np.uint8)  #'uint8')
#        cld_bytes[:,:,:] = cld_packed[:,:,:]
#        bits = np.unpackbits(cld_bytes,axis=0)
#        #
#        # Unpack the flags
#        #
        self.BitFields = self.unpack_Vx_QA(cld_packed, fIn.DayNightFlag == 'Day')
        #
        # Create a fire-noFire-unknown mask. For now, rules are (applied in this order):
        # - cloud and sunglint mean no-data
        # - land means zero for fires above threshold, no-data for smaller ones
        # - actual FRP means fire and its features
        # Coding:
        # np.nan for missing, negative threshold for conditional, actual FRP for fire
        #
        # Get the basic information on the fires - same set as for IS4FIRES v.2.0
        # VIIRS variables:
        # dict_keys(['FP_AdjCloud', 'FP_AdjWater', 'FP_MAD_DT', 'FP_MAD_T4', 'FP_MAD_T5', 
        # 'FP_MeanDT', 'FP_MeanRad13', 'FP_MeanT4', 'FP_MeanT5', 'FP_Rad13', 'FP_SolAzAng',
        # 'FP_SolZenAng', 'FP_T4', 'FP_T5', 'FP_ViewAzAng', 'FP_ViewZenAng', 'FP_WinSize',
        # 'FP_confidence', 'FP_day', 'FP_latitude', 'FP_line', 'FP_longitude', 'FP_power',
        # 'FP_sample', 'algorithm QA', 'fire mask'])
        #
        if fIn.FirePix > 0:
            self.nFires = fIn.FirePix
            self.FP_frp = fIn.variables['FP_power'][:]
            self.FP_line = fIn.variables['FP_line'][:]     # probably index along the track [0:2029]
            self.FP_sample = fIn.variables['FP_sample'][:] # probably, along the scan, [0:1353]
            self.FP_lon = fIn.variables['FP_longitude'][:]
            self.FP_lat = fIn.variables['FP_latitude'][:]
#            print('min-max of lines along line and across sample',
#                  np.min(self.FP_line), np.max(self.FP_line), 
#                 np.min(self.FP_sample), np.max(self.FP_sample))
            self.FP_dS = self.dS[self.FP_sample]   # depends on swath 
            self.FP_dT = self.dT[self.FP_sample]   # also depends on swath
            self.FP_T4 = fIn.variables['FP_T4'][:]
            self.FP_T4b = fIn.variables['FP_MeanT4'][:]
            self.FP_T5 = fIn.variables['FP_T5'][:]
            self.FP_T5b = fIn.variables['FP_MeanT5'][:]
            self.FP_TA = fIn.variables['FP_MeanDT'][:]
            self.FP_satellite = np.array(['S'.encode('utf-8')] * fIn.FirePix, dtype=np.byte)
            if not self.ifSilent: 
                self.log.log('Nbr of fires, FRP: %g, %g MW, %s'  % 
                             (self.FP_frp.shape[0], np.sum(self.FP_frp), str(self.now_UTC)))
        else:
            # No fires
            if not self.ifSilent: self.log.log('No fires')
            self.nFires = 0
        return True


    #=============================================================

    def unpack_Vx_QA(self, arIn, dayNight):
        #
        # 0 not-processed (non-zero QF)     1 bowtie
        # 2 glint                           3 water
        # 4 clouds                          5 clear land
        # 6 unclassified fire pixel         7 low confidence fire pixel
        # 8 nominal confidence fire pixel   9 high confidence fire pixel
        #
        granBits = gb.unpacked_cloud_fields()  # Create the object for storage
        granBits.ifAnalysed = arIn > 2      # processed, not bowtie, not glint
        granBits.QA = arIn * np.nan         # Approximate MODIS QA legend
        granBits.QA[arIn == 4] = 0          # cloudy
        granBits.QA[arIn == 3] = 11         # water
        granBits.QA[arIn >= 5] = 11         # clear land, fires of all kinds 
        granBits.day_night = dayNight
        granBits.sunglint = arIn * np.nan
        granBits.sunglint[np.logical_or(arIn > 4, arIn == 3)] = 0
        granBits.sunglint[arIn == 2] = 1
        granBits.snow = 0
        granBits.land = arIn * np.nan
        granBits.land [arIn == 3] = 0
        granBits.land [arIn >= 5] = 11
        granBits.QA_txt = {0 :'cloudy', 1 :'uncertain', 10 : 'clear_maybe',  11 : 'clear'}
        granBits.land_txt = {0 : 'water', 1 : 'coast', 10 : 'desert', 11 : 'land'}
        return granBits


    #=================================================================

    def geolocation_2_nc(self, ifDownsampled, chFNmOut):
        #
        # Output file will be nc4, with lossless compression
        # For downsampled granule, have to store the downsampled height
        # The same granule can have both restored and downsampled fields. Have to 
        # choose what to store 
        #
        if not os.path.exists(os.path.split(chFNmOut)[0]):
            os.makedirs(os.path.split(chFNmOut)[0])
        # write file
        with nc4.Dataset(chFNmOut + '_tmp', "w", format="NETCDF4") as outf:
            outf.featureType = "VIIRS_geo";
            # dimensions
            if ifDownsampled:
                typeTmp = self.granule_type
            else:
                typeTmp = 'original'
            # store the type
            outf.setncattr('file_type',typeTmp)
            # array shapes
            if typeTmp == 'downsampled_lonlat': shp = self.lon_DS.shape
            elif typeTmp == 'downsampled_cartesian': shp = self.x_DS.shape
            elif typeTmp == 'original' or typeTmp == 'external': shp = self.lon.shape
            else: raise ValueError('Unknown type of granule:' + typeTmp)
            # Dimensions
            outf.createDimension("Along_Track", shp[0])
            outf.createDimension("Along_Scan", shp[1])
            valdims = ("Along_Track","Along_Scan",)
            # What to store:
            if typeTmp == 'downsampled_lonlat':
                arStore = [("Longitude","Longitude","f4","degrees_east",self.lon_DS),
                           ("latitude","latitude",'f4',"degrees_north",self.lat_DS),
                           ('height',"altitude above sea level",'i2','m',self.height_DS)]
            elif typeTmp == 'downsampled_cartesian':
                arStore = [("x","x-Cartezian","f4","km",self.x_DS),
                           ("y","y-Cartesian",'f4',"km",self.y_DS),
                           ("z","z-Cartesian",'f4',"km",self.z_DS),
                           ('height',"altitude above sea level",'i2','m',self.height_DS)]
            elif typeTmp == 'original' or typeTmp == 'external':
                arStore = [("Longitude","Longitude","f4","degrees_east",self.lon),
                           ("latitude","latitude",'f4',"degrees_north",self.lat)]
            else: raise ValueError('Unknown type of granule:' + typeTmp)
            # Store the stuff
            for Nm, NmLong, tp, u, v in arStore:
                var = outf.createVariable(Nm, tp, valdims, zlib=True, complevel=5)
                var.standard_name = Nm
                var.long_name = NmLong
                var.units = u
                var[:,:] = v

#            # longitude
#            lon = outf.createVariable("Longitude","f4", valdims, zlib=True, complevel=5)
#            lon.standard_name = "longitude"
#            lon.long_name = "longitude"
#            lon.units = "degrees_east"
#            # latitude
#            lat = outf.createVariable("Latitude","f4", valdims, zlib=True, complevel=5)
#            lat.standard_name = "latitude"
#            lat.long_name = "latitude"
#            lat.units = "degrees_north"
#            if ifDownsampled:
#                # height field only for downsampled storage
#                height = outf.createVariable('height','i2', valdims, zlib=True, complevel=5)
#                height.standard_name = "height"
#                height.long_name = "altitude above sea level"
#                height.units = "m"
#                lon[:,:] = self.lon_DS
#                lat[:,:] = self.lat_DS
#                height[:,:] = self.height_DS
#            else:
#                lon[:,:] = self.lon
#                lat[:,:] = self.lat
            #
            # Write metadata: groups and attributes of the oroginal file
            #            
            for a in self.dicGlobal_attr.keys():
                if type(self.dicGlobal_attr[a]) == 'str':
                    outf.setncattr(a.replace(' ','_').replace('/','_'), 
                                   self.dicGlobal_attr[a].replace(' ','_').replace('/','_'))
                else:
                    outf.setncattr(a.replace(' ','_').replace('/','_'), self.dicGlobal_attr[a])
            #
            # Mark downsampled field
            #
            if ifDownsampled:
                outf.setncattr('downsampling_factor_along_track', self.DS_factor_track)
                outf.setncattr('downsampling_factor_along_scan', self.DS_factor_scan)
            #
            # groups and group attributes
            #
            for g in self.dicGroups.keys():
                g1 = g.replace(' ','_').replace('/','_')
                outf.createGroup(g1)
                for a in self.dicGroups[g].keys():
                    if type(self.dicGroups[g][a]) == 'str':
                        outf[g1].setncattr(a.replace(' ','_').replace('/','_'),
                                           self.dicGroups[g][a].replace(' ','_').replace('/','_'))
                    else:
                        outf[g1].setncattr(a.replace(' ','_').replace('/','_'), self.dicGroups[g][a])
        #
        # Rename the temporary file. On Windows, unlike Unix, os.rename cannot replace existing file
        # but replace does, albeit it is not an atomic operation there (in Unix it should be atomic)
        #
#        os.replace(chFNmOut + '_tmp', chFNmOut)
        if not self.ifSilent: print('Stored without renaming:', chFNmOut)


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
            if not self.ifSilent: print('Reading FRP from ' + arFNms[0])
            fIn = nc4.Dataset(arFNms[0], 'r')
            self.chFNm_FRP = arFNms[0]
        elif len(arFNms) > 1:
            if not self.ifSilent: 
                self.log.log('Several FRP files satisfy template:' + str(arFNms))
            return False
        else:
            if not self.ifSilent: 
                print('Template:', self.templFRPFiles, self.now_UTC)
                print('Found files: ', glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
            return False
        #
        # Get the IS4FIRES v.2.0 variables: FRP and ix, iy in the swath
        #
        self.FP_frp = fIn.variables['FP_power'][:]
        self.nFires = len(self.FP_frp)
        if self.nFires == 0: return True         # empty granule
        self.FP_line = fIn.variables['FP_line'][:]     # probably index along the track [0:2029]
        self.FP_sample = fIn.variables['FP_sample'][:] # probably, along the swath, [0:1353]
        self.FP_lon = fIn.variables['FP_longitude'][:]
        self.FP_lat = fIn.variables['FP_latitude'][:]
        if not self.ifSilent: 
            print(np.min(self.FP_line), np.max(self.FP_line), 
                  np.min(self.FP_sample), np.max(self.FP_sample))
        self.FP_dS = self.dS[self.FP_sample]
        self.FP_dT = self.dT[self.FP_sample]
        self.FP_T4 = fIn.variables['FP_T21'][:]
        self.FP_T4b = fIn.variables['FP_MeanT21'][:]
        self.FP_T11 = fIn.variables['FP_T31'][:]
        self.FP_T11b = fIn.variables['FP_MeanT31'][:]
        self.FP_TA = fIn.variables['FP_MeanDT'][:]
        self.FP_SolZenAng = fIn.variables['FP_SolZenAng']
        self.satellite = np.array(['S'.encode('utf-8')], dtype=np.byte)
        
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
        #
        fig, axes = mpl.pyplot.subplots(4,2, figsize=(10,16))
        minLon = np.nanmin(self.lon)
        minLat = np.nanmin(self.lat)
        maxLon = np.nanmax(self.lon)
        maxLat = np.nanmax(self.lat)
#        cmap = mpl.pyplot.get_cmap('cool')
        ixAx = 0
        iyAx = 0
        chFires = ' no fires'

        for var, chTxt, cm in [(self.height,'height, m','terrain'),
                               (self.lon,'Lon','cool'),
                               (self.lat,'Lat','cool')]:
            ax = axes[ixAx,iyAx]
            if not self.ifSilent: print('Plotting ', chTxt)
            # draw coastlines, state and country boundaries, edge of map.
            bmap = basemap.Basemap(projection='cyl', resolution='h',    # crude, low, intermediate, high, full 
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
            cs = bmap.scatter(self.lon[::8,::8], self.lat[::8,::8], c=var[::8,::8], 
                              s=1, edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap(cm))
            bmap.colorbar(cs,location='bottom',pad="7%")
#            cbar.set_label(chTxt, fontsize=9)
            ax.set_title(chTxt + 'min=%g, max=%g' % (np.nanmin(var), np.nanmax(var)), fontsize=10)
            if ixAx + iyAx == 0 and self.nFires > 0:
                if not self.ifSilent: print('Plotting FRP')
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
        if self.nFires > 0:
            axes[0,1].yaxis.set_ticklabels([])
        mpl.pyplot.suptitle(os.path.split(chOutFNm)[1].strip('.png') + chFires, fontsize=14)
        if not self.ifSilent: print('Rendering')
        mpl.pyplot.savefig(chOutFNm,dpi=600)
        if not self.ifSilent: print('Saved: ', chOutFNm)
        mpl.pyplot.clf()
        mpl.pyplot.close()


    #===================================================================

    def draw_granule_diff(self, gran2, titles, chOutFNm):
        #
        # Area covered by this swath, with a bit free space around:

        fig, axes = mpl.pyplot.subplots(4,2, figsize=(10,16))
        minLon = np.nanmin(self.lon)
        minLat = np.nanmin(self.lat)
        maxLon = np.nanmax(self.lon)
        maxLat = np.nanmax(self.lat)
        ixAx = 0
        iyAx = 0
        chFires = ' no fires'

        for var, chTxt, cm in [(self.BitFields.ifAnalysed ^ gran2.BitFields.ifAnalysed,
                                'ifAnalysed (1=yes) + FRP, [MW]','cool'),
                               (self.BitFields.QA - gran2.BitFields.QA,
                                'QA: 0=cld,1=?,10=clr?,11=clr','Paired'),
                               (np.ones(shape=self.lon.shape) * (self.BitFields.day_night - 
                                                                 gran2.BitFields.day_night),
                                'day-night','cool'), 
                               (self.BitFields.sunglint - gran2.BitFields.sunglint,'sunglint','cool'),
                               (self.height,'height, m','terrain'),                               
#                               (np.ones(shape=self.lon.shape) * (self.BitFields.snow -
#                                                                 gran2.BitFields.snow),
#                                'snow','cool'),
                               (self.BitFields.land - gran2.BitFields.land,
                                '0=water,1=coast,10=desert,11=land','Paired'),
                               (self.lon - gran2.lon,'Lon','cool'),
                               (self.lat - gran2.lat,'Lat','cool')]:
            ax = axes[ixAx,iyAx]
            if not self.ifSilent: print('Plotting ', chTxt)
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
            cs = bmap.scatter(self.lon[::8,::8], self.lat[::8,::8], c=var[::8,::8], 
                              s=1, edgecolor=None, norm=None, cmap=mpl.pyplot.get_cmap(cm))
            bmap.colorbar(cs,location='bottom',pad="7%")
#            cbar.set_label(chTxt, fontsize=9)
            ax.set_title(chTxt, fontsize=10)
            if ixAx + iyAx == 0 and self.nFires > 0:
                if not self.ifSilent: print('Plotting FRP')
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
        if self.nFires > 0:
            axes[0,1].yaxis.set_ticklabels([])
        mpl.pyplot.suptitle(os.path.split(chOutFNm)[1].strip('.png') + ' '.join(titles) + chFires,
                            fontsize=14)
        if not self.ifSilent: print('Rendering diff')
        mpl.pyplot.savefig(chOutFNm,dpi=200)
        if not self.ifSilent: print('Saved diff: ', chOutFNm)
        mpl.pyplot.clf()
        mpl.pyplot.close()



############################################################################################
############################################################################################
############################################################################################
############################################################################################

def evaluate_downsampling(chFNm_geo375, chFNm_geo750, chFNm_FRP_, chFNmHeight, 
                          tStart, nSteps, chDirLog, ifDS_Cartesian, ifDraw, mpirank):
    #
    # Scans the given time period comparing the downscaling skills for various DS_factor values
    #
    iProcess = 0
    for i in range(nSteps):   # VIIRS has 6 minutes time step

        now = tStart + spp.one_minute * i * 6
        chFNm_Log = os.path.join(chDirLog, 
                                 now.strftime('log_%Y%j_%Y.%m.%d_%H%M') + '_mpi%02g.txt' % mpirank)
        #
        # Reference granule
        #
        chFNm_FRP = 'd:\\data\\satellites\\VIIRS\\VNP14\\2022.05.01\\VNP14IMG.A2022121.1718.001.2022122012909.nc'
        gv_ref = granule_VIIRS('SNPP', now, chFNm_FRP, chFNm_geo375, chFNmHeight, spp.log(chFNm_Log))
        if not gv_ref.pick_granule_data_IS4FIRES_v3_0(): 
            continue
        if ifDraw:
            spp.ensure_directory_MPI(os.path.join(dirMain, 'pics_ref',
                                                  gv_ref.now().strftime('VNP03IMGLL\\%Y.%m.%d')))
            gv_ref.draw_granule(os.path.join(dirMain, 'pics_ref',
                                         gv_ref.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
                                         os.path.split(gv_ref.get_geo_FNm())[1] + '.png'))
        #
        # Downsample the 375 geodata, save to file, and restore
        # Since playing with the same granule, have to reload it for each test
        #
        for iFactor_track, iFactor_scan in [(16,32)]: #, (16,16)]: #, (8,8)]:  #, (4,4), (2,2)]:
            print('375: Handling factor (%gx%g)...' % (iFactor_track, iFactor_scan))
            gv_375 = granule_VIIRS('SNPP', now, chFNm_FRP, chFNm_geo375, chFNmHeight,
                                   spp.log('d:\\tmp\\VIIRS_tst.log'))
            if not gv_375.pick_granule_data_IS4FIRES_v3_0(): continue
            print('Not storing ref granule')
#            gv_375.geolocation_2_nc(False,        # original geodata
#                                    os.path.join(dirMain, 'ref',
#                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
#                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
            t = dt.datetime.now()
#            gv_375.downsample_granule_geo_lonlat(iFactor)
            gv_375.downsample_granule_geo_Cartesian(iFactor_track, iFactor_scan, ifDS_Cartesian)
            gv_375.geolocation_2_nc(True,         # downsampled geodata
                                    os.path.join(dirMain, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan),
                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
#            gv_375.restore_granule_geo_lonlat()
            true_height = gv_375.restore_granule_geo_Cartesian()
            gv_375.geolocation_2_nc(False,
                                    os.path.join(dirMain, 'DS_%02i_%02i_restored' % (iFactor_track, iFactor_scan),
                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
            print('Done down-/up-sampling. Time used:', dt.datetime.now() - t)
            #
            # Compare gv_ref and gv_375: their self.lon and self.lat are directly comparable
            #
            dLon = gv_375.lon - gv_ref.lon
            # take care of a slight shift in different directions from -180
            dLon[dLon > 180] -= 360
            dLon[dLon < -180] += 360
            if np.any(np.abs(dLon) > 100): print('Huge longitude difference')
            if not np.all(np.isfinite(dLon)):
                print('Nans noticed: dLon, gv_375.lon, gv_ref.lon', 
                      np.sum(np.isnan(dLon)), np.sum(np.isnan(gv_375.lon)), np.sum(np.isnan(gv_ref.lon)))
            dLat = gv_375.lat - gv_ref.lat
            if not np.all(np.isfinite(dLat)):
                print('Nans noticed: dLat, gv_375.lat, gv_ref.lat', 
                      np.sum(np.isnan(dLat)), np.sum(np.isnan(gv_375.lat)), np.sum(np.isnan(gv_ref.lat)))
            meanLat = np.mean(gv_ref.lat)
#            lon2km = np.cos(meanLat * spp.degrees_2_radians) * 111.
#            lon2km = np.cos(gv_ref.lat * spp.degrees_2_radians) * 111.
            lat2km = 111.
            dLon_km = dLon * np.cos(gv_ref.lat * spp.degrees_2_radians) * 111.
            if np.any(np.abs(dLon_km) > 10): 
                print('Large longitude-difference (min-max): %g : %g km' % (np.min(dLon_km), np.max(dLon_km)))
            gv_ref.log.log('375: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
                           (iFactor_track, iFactor_scan, np.mean(gv_ref.lon), meanLat, np.max(gv_ref.height))
                           + ' med_Dlon_abs= %gkm, med_Dlat_abs= %gkm, ' % 
                           (np.median(np.abs(dLon_km)), np.median(np.abs(dLat) * lat2km)) 
                           + ' min_Dlon= %gkm, max_Dlon= %gkm, min_Dlat= %gkm, max_Dlat= %gkm,' % 
                           (np.min(dLon_km), np.max(dLon_km), 
                            np.min(dLat) * lat2km, np.max(dLat) * lat2km)
                           + ' 001p_Dlon= %gkm, 001p_Dlat= %gkm, 9999p_Dlon= %gkm, 9999p_Dlat= %gkm' % 
                           (np.percentile(dLon_km,0.01),
                            np.percentile(dLat,0.01) * lat2km,
                            np.percentile(dLon_km,99.99),
                            np.percentile(dLat,99.99) * lat2km))
            if not (np.all(np.isfinite(dLon)) and np.all(np.isfinite(dLat))):
                gv_ref.log.log('NAN 375: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
                               (iFactor_track, iFactor_scan, np.mean(gv_ref.lon), meanLat, np.max(gv_ref.height))
                               + ' n_dLon=%i, n_gv_375.lon=%i, n_gv_ref.lon=%i' %
                               (np.sum(np.isnan(dLon)), np.sum(np.isnan(gv_375.lon)), np.sum(np.isnan(gv_ref.lon)))
                               + ' n_dLat=%i, n_gv_375.lat=%i, n_gv_ref.lat=%i' %
                               (np.sum(np.isnan(dLat)), np.sum(np.isnan(gv_375.lat)), np.sum(np.isnan(gv_ref.lat)))
                               )
            # Draw
            #
            if ifDraw:
                spp.ensure_directory_MPI(os.path.join(dirMain, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_375.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_375.draw_granule(os.path.join(dirMain, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                             gv_375.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                             os.path.split(gv_375.get_geo_FNm())[1] + '.png'))
                spp.ensure_directory_MPI(os.path.join(dirMain, 'DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_375.draw_granule_diff(gv_ref, ['375m','DS_%g_%g' % (iFactor_track, iFactor_scan)],
                                     os.path.join(dirMain, 'DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                  os.path.split(gv_ref.get_geo_FNm())[1] + '.png'))
            
        #
        # Downsample the 750 geodata, save to file, and restore the geodata
        #
        for iFactor_track, iFactor_scan in []:  #(16,32), (16,16), (8,8), (4,4), (2,2)]:
            print('750: Handling factor %g...' % iFactor)
            gv_750 = granule_VIIRS('SNPP', now, chFNm_FRP, chFNm_geo750, chFNmHeight,
                                   spp.log('d:\\tmp\\VIIRS_tst.log'))
            if not gv_750.pick_granule_data_IS4FIRES_v3_0(): continue
            # Store reference granules
            gv_750.geolocation_2_nc(False,        # Full geodata 
                                    os.path.join(dirMain, 'g750_ref',
                                                 gv_750.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                 os.path.split(gv_750.get_geo_FNm())[1] + '.nc4'))
            t = dt.datetime.now()
#            gv_750.downsample_granule_geo_lonlat(iFactor)
            gv_750.downsample_granule_geo_Cartesian(iFactor_track, iFactor_scan)
            gv_750.geolocation_2_nc(True,         # downsampled geodata
                                    os.path.join(dirMain, 'g750_DS_%02i_%02i' % (iFactor_track, iFactor_scan),
                                                 gv_750.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                 os.path.split(gv_750.get_geo_FNm())[1] + '.nc4'))
#            gv_750.restore_granule_geo_lonlat()
            gv_750.restore_granule_geo_Cartesian()
            gv_750.geolocation_2_nc(False,         # Full geodata
                                    os.path.join(dirMain, 'g750_DS_%02i_%02i_restored' % (iFactor_track, iFactor_scan),
                                                 gv_750.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                 os.path.split(gv_750.get_geo_FNm())[1] + '.nc4'))
            print('Done. Time used:', dt.datetime.now() - t)
            #
            # Compare gv_ref and gv_750: their self.lon and self.lat are directly comparable
            #
            dLon = gv_750.lon - gv_ref.lon
            if not np.all(np.isfinite(dLon)):
                print('Nans noticed: dLon, gv_750.lon, gv_ref.lon', 
                      np.sum(np.isnan(dLon)), np.sum(np.isnan(gv_750.lon)), np.sum(np.isnan(gv_ref.lon)))
            # take care of a slight shift in different directions from -180
            dLon[dLon > 180] -= 360
            dLon[dLon < -180] += 360
            if np.any(dLon > 100):
                print('Huge longitude difference')
            dLat = gv_750.lat - gv_ref.lat
            if not np.all(np.isfinite(dLat)):
                print('Nans noticed: dLat, gv_750.lat, gv_ref.lat', 
                      np.sum(np.isnan(dLat)), np.sum(np.isnan(gv_750.lat)), np.sum(np.isnan(gv_ref.lat)))
            meanLat = np.mean(gv_ref.lat)
#            lon2km = np.cos(meanLat * spp.degrees_2_radians) * 111.
#            lon2km = np.cos(gv_ref.lat * spp.degrees_2_radians) * 111.
            lat2km = 111.
            dLon_km = dLon * np.cos(gv_ref.lat * spp.degrees_2_radians) * 111.
            gv_ref.log.log('750: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
                           (iFactor_track, iFactor_scan, np.mean(gv_ref.lon), meanLat, np.max(gv_ref.height))
                           + ' med_Dlon_abs= %gkm, med_Dlat_abs= %gkm, ' % 
                           (np.median(np.abs(dLon_km)), np.median(np.abs(dLat)) * lat2km) 
                           + ' min_Dlon= %gkm, max_Dlon= %gkm, min_Dlat= %gkm, max_Dlat= %gkm,' % 
                           (np.min(dLon_km), np.max(dLon_km), 
                            np.min(dLat) * lat2km, np.max(dLat) * lat2km)
                           + ' 001p_Dlon= %gkm, 001p_Dlat= %gkm, 9999p_Dlon= %gkm, 9999p_Dlat= %gkm' % 
                           (np.percentile(dLon_km,0.01),
                            np.percentile(dLat,0.01) * lat2km,
                            np.percentile(dLon_km,99.99),
                            np.percentile(dLat,99.99) * lat2km))
            if not (np.all(np.isfinite(dLon)) and np.all(np.isfinite(dLat))):
                gv_ref.log.log('NAN 750: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
                               (iFactor_track, iFactor_scan, np.mean(gv_ref.lon), meanLat, np.max(gv_ref.height))
                               + ' n_dLon=%i, n_gv_750.lon=%i, n_gv_ref.lon=%i' %
                               (np.sum(np.isnan(dLon)), np.sum(np.isnan(gv_750.lon)), np.sum(np.isnan(gv_ref.lon)))
                               + ' n_dLat=%i, n_gv_750.lat=%i, n_gv_ref.lat=%i' %
                               (np.sum(np.isnan(dLat)), np.sum(np.isnan(gv_750.lat)), np.sum(np.isnan(gv_ref.lat)))
                               )
            # Draw
            #
            if ifDraw:
                spp.ensure_directory_MPI(os.path.join(dirMain, 'g750_DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_750.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_750.draw_granule(os.path.join(dirMain, 'g750_DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                             gv_750.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                             os.path.split(gv_750.get_geo_FNm())[1] + '.png'))
                spp.ensure_directory_MPI(os.path.join(dirMain, 'g750_DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_750.draw_granule_diff(gv_ref, ['750m','750_DS_%g_%g' % (iFactor_track, iFactor_scan)],
                                     os.path.join(dirMain, 'g750_DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                  os.path.split(gv_ref.get_geo_FNm())[1] + '.png'))


############################################################################

def get_IMG_height_calculate(chFNm_geo375, chFNm_FRP, tStart, nSteps, chOutDir):
    #
    # Scans over the given time period and collects all hits of the height field in
    # different granules. In the end, obtains the global high-resolution height field 
    #
    ifMediumRes = False
    # reserve the space
    if ifTest_4height:
        # testing: 10 km
        nx = 3600
        ny = 1800
        chRes = '10km'
    elif ifMediumRes:
        # for real: 1000 m
        nx = 36000
        ny = 18000
        chRes = '1km'
    else:
        # for big real: 500 m
        nx = 72000
        ny = 36000
        chRes = '500m'
    deg2idx = nx/360
    height = np.zeros(shape = (ny, nx), dtype=np.float32)
    cnt = np.zeros(shape = (ny, nx), dtype=np.uint32)
    if not os.path.exists(chOutDir): os.makedirs(chOutDir)
    #
    # Cycle over time
    #
    for i in range(nSteps):   # VIIRS has 6 minutes time step
        now = tStart + spp.one_minute * i * 6
        chFNm_Log = os.path.join(chOutDir, 
                                 now.strftime('log_IMG_height_%Y%j_%Y.%m.%d_%H%M') +'.txt')
        # IMG granule
        #
        gv = granule_VIIRS('SNPP', now, chFNm_FRP, chFNm_geo375, '', spp.log(chFNm_Log))
        if not gv.pick_granule_data_IS4FIRES_v3_0(): continue
        #
        # Go over the lon-lat fields adding height and counter
        #
        idx_X = ((gv.lon + 180) * deg2idx).astype(np.int32)
        idx_Y = ((gv.lat + 90) * deg2idx).astype(np.int32)
        height[idx_Y, idx_X] += gv.height[:,:]
        cnt[idx_Y, idx_X] += 1
    #
    # Normalize and store
    #
    height /= cnt
    height[cnt==0] = np.nan
    
    return (height, chRes, deg2idx, chOutDir, now)

#===============================================================================

def get_IMG_height_store(height, chRes, deg2idx, chOutDir, now):
#
    # Store the fields
    #
    with nc4.Dataset(os.path.join(chOutDir, 'height_%s.nc4' % chRes), "w", format="NETCDF4") as outf:
        outf.featureType = "VIIRS_height";
        # dimensions

        outf.createDimension("longitude", height.shape[0])
        lon = outf.createVariable('longitude', 'f', ('longitude',))
        lon[:] = np.arange(-180, 180, 1./deg2idx)
        lon.units = 'degrees_east'
        lon.axis = "X"
        
        outf.createDimension("latitude", height.shape[1])
        lat = outf.createVariable('latitude', 'f', ('latitude',))
        lat[:] = np.arange(-90,90, 1/deg2idx)
        lat.units = 'degrees_north'
        lat.axis = "Y"

        outf.createDimension("time", None)
        t = outf.createVariable("time","i4",("time",))
        t.standard_name="time"
        t.long_name="last time of the height field"
        t.calendar="standard"
        t.units = now.strftime("minutes since %Y-%m-%d %H:%M:%S UTC")
        t[:] = [0]
        t.timezone = ''

        # height
        hgt = outf.createVariable("relief_height","i2",("latitude","longitude",), zlib=True,
                                  complevel=5)
        hgt.standard_name = "relief_height"
        hgt.long_name = "relief height"
        hgt.units = "m"
        hgt[:,:] = np.round(height).astype(np.int16)
#        if ifTest:
#            # counter
#            counter = outf.createVariable("counter","i4",("longitude","latitude",), zlib=True,
#                                          complevel=5)
#            counter.standard_name = "counter"
#            counter.long_name = "Number of VIIRS pixels"
#            counter.units = ""
#            counter[:,:] = cnt

#===============================================================================

def get_IMG_height_draw(height, chRes, chOutDir):
    #
    # Draw
    #
    fig, axes = mpl.pyplot.subplots(2,1, figsize=(10,16))
    for var, chTitle, ax, cmap in [(height, 'Height ASL, m', axes[0], 'terrain')]: #,
#                                   (cnt, 'N-hits', axes[1], 'jet')]:
        # draw coastlines, state and country boundaries, edge of map.
        bmap = basemap.Basemap(projection='cyl', resolution='l',    # crude, low, intermediate, high, full 
                               llcrnrlon = -180, urcrnrlat = 90,
                               urcrnrlon = 180, llcrnrlat = -90, ax=ax)
        bmap.drawcoastlines(linewidth=0.5)
        bmap.drawcountries(linewidth=0.4)
        # draw parallels and meridians
        bmap.drawmeridians(np.arange(-180, 180, 30), labels=[0,0,0,1], fontsize=10)
        bmap.drawparallels(np.arange(-90, 90, 30), labels=[1,0,0,0],fontsize=10)
        # draw filled contours
        cs = bmap.imshow(var, norm=None, cmap=mpl.pyplot.get_cmap(cmap))
        bmap.colorbar(cs,location='bottom',pad="7%")
        ax.set_title(chTitle, fontsize=10)
    mpl.pyplot.savefig(os.path.join(chOutDir,'height_%s.png' % chRes),dpi=100)
    mpl.pyplot.clf()
    mpl.pyplot.close()


#########################################################################

def log_skill_summary(chFNmTempl):
    #
    # Reads the bunch of log files and summarisesthe downsampling 
    # accuracy skills
    #
    dicSkills = {}
    for chFIn in glob.glob(chFNmTempl):
        print(chFIn)
        for line in open(chFIn, 'r'):
            if not 'DS_factor' in line: continue
            flds = line.replace('km 0','km, 0').split(',')
            #
            # 375: DS_factor= (16,16), Lon~ -149.718, lat~ -47.7197, height~ 0m, mean_Dlon_abs= 0.0115072km, mean_Dlat_abs= 0.0031417km,  min_Dlon~ -1.30811km, max_Dlon~ 1.6659km, min_Dlat~ -0.598309km, max_Dlat~ 0.118137km
            # 375: DS_factor= (4,4), Lon= 134.536, lat= 27.0489, height= 26.6631m, med_Dlon_abs= 0.00150846km, med_Dlat_abs= 0.000211716km,  min_Dlon= -6.00142km, max_Dlon= 4.38963km, min_Dlat= -0.382994km, max_Dlat= 0.487158km 001p_DLon= -1.32748km, 001p_DLat= -0.0897536km, 9999p_DLon= 1.34404km, 9999p_DLat= 0.0899422km
            #
            f1 = flds[0].split(':')
            resStart = int(f1[0])
            DSf  = f1[1].split('=')[1].strip()  # string
            try: dicSkills[resStart][DSf]
            except: 
                try: dicSkills[resStart][DSf] = {}
                except: dicSkills[resStart] = {DSf:{}}
            for fld in flds[1:]:
                if '=' in fld: dicItem = fld.split('=')
                elif '~' in fld: dicItem = fld.split('~')
                try: dicSkills[resStart][DSf][dicItem[0].strip()].append(
                    float(dicItem[1].strip('\n').strip('m').strip('km')))
                except: 
                    try:
                        dicSkills[resStart][DSf][dicItem[0].strip()] = [
                                float(dicItem[1].strip('\n').strip('m').strip('km'))]
                    except:
                        print('??')
    print(list(dicSkills.keys()))
    #
    # Need: median performance, upper and lower percentiles, dependence on height abd lat
    # First, reshuffle
    #
    r0 = list(dicSkills.keys())[0]
    DSf0 = list(dicSkills[r0].keys())[0]
    chParams = list(dicSkills[r0][DSf0].keys())
    chParams.pop(chParams.index('lon'))
    chParams.pop(chParams.index('lat'))
    try: 
        chParams.pop(chParams.index('heightMAX'))
        height = dicSkills[r0][DSf0]['heightMAX']
    except: 
        print('Obsolete log file format')
        chParams.pop(chParams.index('height'))
        height = dicSkills[r0][DSf0]['height']
    lat = dicSkills[r0][DSf0]['lat']
    ParamGroups = [('med_Dlon_abs',None), ('med_Dlat_abs',None),
                   ('min_Dlon','max_Dlon'),('min_Dlat','max_Dlat'),
                   ('001p_Dlon','9999p_Dlon'),('001p_Dlat','9999p_Dlat')]
    print('ref_res DSf', chParams, ', ==>>km')
    dicSum = {}
    arPar = {}
    for r in sorted(list(dicSkills.keys())):
        dicSum[r] = {}
        arPar[r] = {}
        for DSf in sorted(list(dicSkills[r].keys())):
            dicSum[r][DSf] = {'med_Dlon_abs': np.median(dicSkills[r][DSf]['med_Dlon_abs'])}
            dicSum[r][DSf]['med_Dlat_abs'] = np.median(dicSkills[r][DSf]['med_Dlat_abs'])
            dicSum[r][DSf]['min_Dlon'] = np.min(dicSkills[r][DSf]['min_Dlon'])
            dicSum[r][DSf]['max_Dlon'] = np.max(dicSkills[r][DSf]['max_Dlon'])
            dicSum[r][DSf]['min_Dlat'] = np.min(dicSkills[r][DSf]['min_Dlat'])
            dicSum[r][DSf]['max_Dlat'] = np.max(dicSkills[r][DSf]['max_Dlat'])
            dicSum[r][DSf]['001p_Dlon'] = np.min(dicSkills[r][DSf]['001p_Dlon'])
            dicSum[r][DSf]['001p_Dlat'] = np.min(dicSkills[r][DSf]['001p_Dlat'])
            dicSum[r][DSf]['9999p_Dlon'] = np.max(dicSkills[r][DSf]['9999p_Dlon'])
            dicSum[r][DSf]['9999p_Dlat'] = np.max(dicSkills[r][DSf]['9999p_Dlat'])
            arPar[r][DSf] = {}
            for p in chParams:
                arPar[r][DSf][p] = np.array(dicSkills[r][DSf][p])
            print(r,DSf,', '.join('%g' % dicSum[r][DSf][k] for k in chParams))
            #
            # Draw the statistics vs height and latiitude
            #
            for ext, x, col in [('_H',height,lat),('_lat',lat,height)]:
                fig = mpl.pyplot.figure(constrained_layout=True, figsize=(10,12))  #(10,18))
                gs = fig.add_gridspec(3,2) #(5,2)
                ix = 0
                iy = 0
                for p in ParamGroups:  #chParams:
                    ax = fig.add_subplot(gs[iy,ix])
                    plot = ax.scatter(x, np.array(dicSkills[r][DSf][p[0]]), 
                                      label=p[0], s=8, c=col, cmap='gnuplot_r')
                    if p[1]:
                        plot2 = ax.scatter(x, np.array(dicSkills[r][DSf][p[1]]), 
                                           label=p[1], s=8, c=col, marker = '^', cmap='gnuplot_r')
                        ax.set_ylabel(',  '.join(p) + ', km, color:' + {'_H':' deg lat', '_lat':' m heightMAX'}[ext])
                        ax.set_title(',  '.join(p))
                        ax.legend()
                    else:
                        ax.set_title(p[0])
                        ax.set_ylabel(p[0] + ', km, color:' + {'_H':' deg lat', '_lat':' m heightMAX'}[ext])
                    ax.set_xlabel(ext[1:] + {'_H':', m', '_lat':', deg'}[ext])
                    mpl.pyplot.colorbar(plot)
                    ax.grid()
                    ix += 1
                    if ix > 1: 
                        ix = 0
                        iy += 1
                mpl.pyplot.savefig(os.path.join(os.path.split(chFIn)[0],
                                                'par_vs%s_%g_%s.png' % (ext,r,DSf)), dpi=200)
                fig.clf()
                mpl.pyplot.close()
                print(os.path.join(os.path.split(chFIn)[0],
                                   'par_vs%s_%g_%s.png' % (ext,r,DSf)))
    #
    # Reformat the data abd draw integral bar charts
    #
    fig = mpl.pyplot.figure(constrained_layout=True, figsize=(10,18))
    gs = fig.add_gridspec(5,2)
    ix = 0
    iy = 0
    for p in chParams:
        ax = fig.add_subplot(gs[iy,ix])
        dat = {}
        for r in sorted(list(dicSkills.keys())):
            dat[r] = []
            for DSf in sorted(list(dicSkills[r].keys())):
                dat[r].append(dicSum[r][DSf][p])
        drawer.bar_plot(ax, dat, data_stdev=None, colors=None, 
                        total_width=0.8, single_width=1, 
                        legend=(True, 'upper right', 9, None),   # if needed, location, fontsize 
                        group_names=sorted(list(dicSkills[r].keys())))
        ax.set_title(p)
        ax.set_xlabel('Downscaling factor')
        ax.set_ylabel('km')
        ix += 1
        if ix > 1: 
            ix = 0
            iy += 1
    mpl.pyplot.savefig(os.path.join(os.path.split(chFIn)[0], 'par_total_bar.png'), dpi=200)
    fig.clf()
    mpl.pyplot.close()

    return (dicSkills, dicSum)


########################################################################

def check_pixel_fit(chLUT_FNm):
        # Roux's fit, not really needed
#        dS_LUT_fit, dT_LUT_fit = LUT.viirs_pixel_size_LUT()
        # Initial grid cell sizes
        with open(chLUT_FNm,'r') as fIn:
            i_pix = []
            dS_pix = []
            dT_pix = []
            thet_pix = []
            for line in fIn:
                if line.startswith('#'): continue
                flds = line.split(',')
                i_pix.append(int(flds[0]))
                dS_pix.append(float(flds[1]))
                dT_pix.append(float(flds[2]))
                thet_pix.append(float(flds[4]))  # yes, 3 is the pixel area
                if abs(dS_pix[-1] * dT_pix[-1] / float(flds[3]) - 1) > 1e-3:
                    print('Wrong line', line)
                    sys.exit()
        dS_LUT = np.array(dS_pix)
        dT_LUT = np.array(dT_pix)
        theta_LUT = np.array(thet_pix)
        t2 = theta_LUT[:theta_LUT.size//2+1]
        t2[theta_LUT.size//2] = 0
        t3 = (t2[1:] + t2[:-1])/2
        theta_LUT = np.concatenate((t3,t3[::-1]))
        
#        theta_LUT -= 0.0266783 / 2.0   # np.min(theta_LUT)
                
        # ATBD of MODIS applied to VIIRS by modifying the flight geometry
        gran = granule_VIIRS('VIIRS', dt.datetime.utcnow(), '', '', '', 
                             spp.log(os.path.join(dirMain,'fit_pixel_size.log')))
        gran.log.log('\ndS: i, dS_LUT, dS_fla, DS_LUT-dS_fla, rel_error')
        for i in range(gran.N_375):
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
        axes[0].plot(range(gran.N_375), gran.dS, label='dS', c='blue') #, dS_LUT - gran.dS)
        ax1.plot(range(gran.N_375), dS_LUT - gran.dS, label='diff', c='red')
        axes[0].set_ylabel('dS, km', color='blue')
        ax1.set_ylabel('dS_LUT - dS, km', color='red')
        # dT
        ax2 = axes[1].twinx()
        axes[1].plot(range(gran.N_375), gran.dT, label='dT', c='blue') #, dS_LUT - gran.dS)
#        axes[1].plot(range(gran.N_375), dT_LUT+0.1, label='dT_LUT', c='green') #, dS_LUT - gran.dS)
        ax2.plot(range(gran.N_375), dT_LUT - gran.dT, label='diff_dT', c='red')
        axes[1].set_ylabel('dT, km', color='blue')
        ax2.set_ylabel('dT_LUT - dT, km', color='red')
        # theta
        ax3 = axes[2].twinx()
        axes[2].plot(range(gran.N_375), gran.theta, label='theta', c='blue') #, dS_LUT - gran.dS)
        ax3.plot(range(gran.N_375), theta_LUT * np.pi / 180.0 - np.abs(gran.theta), label='diff', c='red')
        axes[2].set_ylabel('theta, rad', color='blue')
        ax3.set_ylabel('theta_LUT - theta, rad', color='red')
        mpl.pyplot.savefig(os.path.join(dirMain,'pixel_size_fit.png'), bbox_inches='tight', dpi=200)
#        mpl.pyplot.show()

##########################################################################
#
# Open-up the dS, dT, and theta for external use without creating granule
#
granSLSTR_ref = granule_SLSTR()
dS = granSLSTR_ref.dS
dT = granSLSTR_ref.dT
theta = granSLSTR_ref.theta


##########################################################################
##########################################################################
##########################################################################
##########################################################################

if __name__ == '__main__':
    #
    # test the SLSTR data
    #
    dirMain = 'd:\\data\\satellites\\SLSTR\\NTC_not_time_critical'
    chDataDirNm = 'S3A_SL_2_FRP____%Y%m%dT%H%M*'
#    'S3A_SL_2_FRP____20230304T065159_20230304T065459_20230305T153712_0179_096_134_3780_PS1_O_NT_004.SEN3'
    chFNm_FRP = os.path.join(dirMain, chDataDirNm, 'FRP_in.nc')
    chFNm_geo = os.path.join(dirMain, chDataDirNm, 'geodetic_in.nc')
    chFNm_height = os.path.join(dirMain, 'att1_16_16_basic\\height\\height_1km.nc4')
    timeGranule = dt.datetime(2023,3,4,6,51,59)
    lstDirFullNm = glob.glob(os.path.join(dirMain,timeGranule.strftime(chDataDirNm)))
    if len(lstDirFullNm) > 1: raise ValueError('More than one dir for %s' % timeGranule.strftime(chDataDirNm))
    elif len(lstDirFullNm) < 1: raise ValueError('Zero dirs for %s' % timeGranule.strftime(chDataDirNm))
    else: chDirFullNm = lstDirFullNm[0]
    caseNm = os.path.split(chDirFullNm)[-1]
    
    ifDraw = False
    if_DS_Cartesian = True
    
    ifGeometry = True
    ifEvaluate_DS = False
    ifLogSummary = False
    ifFitPixel_size = False
    #
    # Draw the basic features of the satellite geometry
    #
    if ifGeometry:
        granule = granule_SLSTR(now_UTC=timeGranule)
        granule.pick_granule_data_IS4FIRES_v3_0()
        granule.draw_granule(os.path.join(chDirFullNm, 'pics','granule.png'))
        granule.geolocation_2_nc(False, os.path.join(dirMain,'geo_%s.nc4' % caseNm))  # ifDownsampled, name
        granule.draw_pixel_size(os.path.join(dirMain, 'SLSTR_pixel.png'))
        sys.exit() 
       
    #
    # Evaluate the skills of the downsampling procedure
    #
    if ifEvaluate_DS:
        spp.ensure_directory_MPI(os.path.join(dirMain,'log')) 
        evaluate_downsampling(chFNm_geo, chFNm_geo, chFNm_FRP, chFNm_height,
                              dt.datetime(2022,3,7,2,36), 1,   #47, 
                              os.path.join(dirMain,'log'), if_DS_Cartesian, ifDraw)
    #
    # Get the global IMG height field
    #
    if ifGetGlobal_Height: 
        height,chRes,deg2idx,chOutDir,now = get_IMG_height_calculate(chFNm_geo, chFNm_FRP, 
                                                                     dt.datetime(2022,5,8), 240,  # 240
                                                                     os.path.join(dirMain,'height'))
        get_IMG_height_store(height, chRes, deg2idx, chOutDir, now)
#        get_IMG_height_draw(height,chRes,chOutDir)

    #
    # Summary for downsampling skills
    #
    if ifLogSummary:
#        log_skill_summary('d:\\data\\satellites\\VIIRS\\logs\\laptop_att1\\*.txt')
#        log_skill_summary('d:\\data\\satellites\\VIIRS\\att2_16_32_DSlonlat\\log\\*.txt')
        log_skill_summary('j:\\data\\satellites\\VIIRS\\log\\*.txt')

    #
    # Checking the quality of the fit for the approximate formula for dS and dT
    # Comparison is done with regard to MODIS / VIIRS lookup tables from PixSizeLUT.py
    #
    if ifFitPixel_size:
        check_pixel_fit('d:\\data\\satellites\\VIIRS\\VIIRS_LUT.txt')
