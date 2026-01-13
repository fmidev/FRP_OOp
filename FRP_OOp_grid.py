'''

Observation Operator for Fire Radiative Poewr products of low-orbit satellites.
Currently available coefficients are for MODIS and VIIRS
Supports both pixel- and grid-wise representation of the fire data.

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

@author: Mikhail Sofiev, Finnish Meteorological Institute
'''


import os, sys, numpy as np, datetime as dt
from toolbox import silamfile, gridtools, supplementary as spp
import granule_MODIS as MxD_gran
import process_satellites as proc_sat
from support import suntime
import FRP_OOp_pixel


########################################################################################################
########################################################################################################
#
# FRP observation operator
# Made for low-orbit satellites, probably same works for geostationary ones, just parameters
# differ
# Made generic: one can supply a single value of FRP or an array of any dimension (should be the same
# for all input data, of course).
# A set of supplementary functions allows calculating the locations and geometry of pixels 
# for known MODIS overpasses.
# 
########################################################################################################
########################################################################################################

class FRP_observation_operator_grid(FRP_OOp_pixel.FRP_observation_operator_pixel):
    
    #------------------------------------------------------------------------
    #
    # Use it to hard-code the parameters of the ObsOp
    # There are two parameters dependent on pixel size: detection limit and sigmoid rate
    # Rest is global constants like 0.0 and 1.0. They are hard-coded.
    # The operator follows the notations in the paper Sofiev, 2022.
    # 
    def __init__(self, chFRP_FNmTempl, chCld_FNmTempl, chInstrument, grid_def, log):
        #
        # Instruments known this-far: MODIS, VIIRS
        #
        self.chInstr = chInstrument
        self.sources2use = {'MODIS':['MOD','MYD'],
                            'VIIRS':['VNP14IMG','VJ114IMG']}[chInstrument]
        self.granule_tStep = {'MODIS':spp.one_minute * 5,
                              'VIIRS':spp.one_minute * 5}[chInstrument]
        self.chFRP_FNmTempl = chFRP_FNmTempl
        self.chCld_FNmTempl = chCld_FNmTempl
        self.grid_def = grid_def
        self.log = log
        #
        # Main Fire Fraction MFF 2-D fitting parameters
        # Universal fitting, all LUs
        #
        self.P0 = 8.65  # MW, 
        self.P1 = 10.     # MW, 
        self.P2 = 143000.   # MW, 
        self.k1 = 0.421
        self.k2 = 3
        self.d0 = 0.05   # degree
        self.NOM = (self.P1 / self.P0 - 1.)**self.k1
        self.resol_factor = np.exp(-self.d0 / np.sqrt(self.grid_def[0].dx * self.grid_def[0].dy))
        #
        # for gridded cloudy DL, need two more parameters
        #
        self.DL_cld = 10000  # MW: something really big
        self.sigma_cld = 0.1      # width of transition from no-cloud to full-cloud
        self.srate_mean = {'MODIS':(1.2, 0.5),     # (night, day)
                           'VIIRS':(15, 1.5)}[chInstrument]
        #
        # For the period of MODIS observations, we can have exact pattern, need history
        # For the period outside the satellite observations or if we do not need high precision
        # make a "typical" clear-sky detection limit and slope
        #
        if chFRP_FNmTempl is None: 
            self.chSathist = None
            self.DL_mean = self.detection_limit(np.ones(shape=(2)) * self.mean_pixel_size, [True,False])
            self.srate_mean = self.slope_rate(np.ones(shape=(2)) * self.mean_pixel_size, [True,False])
#        else:
#            self.MODIShist = silamfile.SilamNCFile(chMODIShist_FNmTempl)
#            self.MODISreader_DL = self.MODIShist.get_reader('detection_limit')
#            self.MODISreader_noData = self.MODIShist.get_reader('missing_cells')
#            self.tStart = self.MODIShist.t()[0]
#            self.tEnd = self.MODIShist.t()[-1]


#    #-----------------------------------------------------------------------
#    # CORE functions
#    #-----------------------------------------------------------------------
#    #
#    # The core and the final step of the observation operator
#    # Observed FRP from modelled FRP using the detection limit and rate of the signoid
#    #
#    def observe(self, arFRP_mdl, arDL, arRate):
#        return arFRP_mdl / (1. + np.exp(-arRate * (arFRP_mdl - arDL)))
#
#
#    #-----------------------------------------------------------------------
#    #
#    # Core of the detection limit, works for any array
#    #
#    def detection_limit(self, pixel_size, ifNight):
#        DL = self.DL_day_slope * pixel_size + self.DL_day_intercept
#        DL[ifNight] = self.DL_night_slope * pixel_size[ifNight] + self.DL_night_intercept
#        return DL.astype(np.float32)
#    #
#    # and the slope rate
#    #
#    def slope_rate(self, pixel_size, ifNight):
#        srate = self.day_r_0 + self.day_r_1 / (pixel_size - self.day_A_0)
#        srate[ifNight] = self.night_r_0 + self.night_r_1 / (pixel_size[ifNight] - self.night_A_0)
#        return srate.astype(np.float32)


    #-----------------------------------------------------------------------
    #
    # The main interface function: observe map
    # One can give AOD at 4 micrometers and cloud mask
    #
    def observe_map(self, mdlFRP, dayUTC, mapAOD_4um=0.0, mapClouds=None):
        #
        # Main-Fire fraction, parameter gamma in the paper. The only array is the FRP
        #
        MFF = 1 - np.exp(mdlFRP/self.P2) * (1. - self.NOM / 
                                            ((mdlFRP / self.P0 - 1)**self.k1 * self.resol_factor))**self.k2
        #
        # Do the whole day at once
        #
        sza = np.zeros(shape=MFF.shape)
        for hour in range(24):
            sza[hour,:,:] = spp.solar_zenith_angle(self.grid_def[0].x()[:,None], self.grid_def[0].y()[None,:], 
                                                   dayUTC.timetuple().tm_yday, hour, 0).T
        ifNight = np.abs(sza) > 90
        #
        # Create the data structure
        #
        mapTmp = MxDproc.daily_maps(self.grid_def, dayUTC, None,  #cloud_threshold,
                                    False, 24, None, self.log) # ifLocalTime=None, nTimes, LU, log
        #
        # Fil-in the data
        #
        mapTmp.Fill_daily_maps(self.chFRP_FNmTempl, self.chCld_FNmTempl, self.sources2use, 
                               self.granule_tStep, False) #ifDrawGranules
    #        arSun = suntime.Sun(self.grid.y(), self.grid.x())   # initiate the object (lat, lon)
    #        sunrise = arSun.get_sunrise_time(dayUTC)
    #        ifNight = np.logical_or(arSun.get_sunrise_time(dayUTC) > dayUTC,
    #                                arSun.get_sunset_time(dayUTC) < dayUTC)
        #
        # The satellite daily set of maps is loaded. The clear-sky detection limit is there,
        # satellite cloudiness too, but we might have our own clouds.
        #
        if mapClouds is None:
            DL_cld = mapTmp.mapDetectLim / MFF + self.DL_cld / (1. + np.exp((1. - mapTmp.mapNoData) / 
                                                                            self.sigma_cld))
        else:  
            DL_cld = mapTmp.mapDetectLim / MFF + self.DL_cld / (1. + np.exp((1. - mapClouds) / 
                                                                            self.sigma_cld))

        P_obs = mdlFRP / (1. + np.exp(-self.srate_mean[1] * (mdlFRP - DL_cld)))  # use daytime slope
        P_obs[ifNight] = mdlFRP[ifNight] / (1. + np.exp(-self.srate_mean[0] * (mdlFRP[ifNight] - 
                                                                               DL_cld[ifNight])))
        P_obs *= np.exp(-mapAOD_4um)
        return (P_obs, DL_cld, mapTmp.mapDetectLim, mapTmp.mapNoData)
    
    
    #-----------------------------------------------------------------------
    #
    # A grid cell observation operator.
    # Determine the grid-cell detection limit, a grid-cell sigmoid slope then call the 
    # self.observe function
    #
    def observe_grid_cells(self, arFRP_mdl, grid, lons, lats, hourUTC, ifDetailed, arMiss=None):
        #
        # No matter what, need to know the day-night split
        #
        arSun = suntime.Sun(lons, lats)   # initiate the object
        ifNight = np.logical_or(arSun.get_sunrise_time(hourUTC) > hourUTC,
                                arSun.get_sunset_time(hourUTC) < hourUTC)
        #
        # Detection limit for the grid cell. First, clearsky
        #
        if ifDetailed:
            #
            # Detailed DL means an explicit mean of the satellite piexl-level detection limit
            # Makes sense for high-resolution grids, for instance
            # Expensive: have to go back to the satellite granules and get the orbit parameters
            #
            if arMiss is None:
                DL_clrsky, srate, arMiss = self.DL_gridded_clearsky(arFRP_mdl, grid, lons, lats,
                                                                    hourUTC)
            else:
                DL_clrsky, srate = self.DL_gridded_clearsky(arFRP_mdl, grid, lons, lats, hourUTC)
        else:
            #
            # If grid is much coarser than the pixel _and_ comparable with the width of the swath,
            # one can take mean DL over the swath
            #
            DL_clrsky = np.where(ifNight, self.DL_mean[0], self.DL_mean[1])
            srate = np.where(ifNight, self.srate_mean[0], self.srate_mean[1])
        #
        # Now, add cloudiness, which is the arMiss - the gridded field, either obtained from 
        # the satellite or given from above
        #
        return self.observe(arFRP_mdl,
                            DL_clrsky + self.DL_full_cld / (1.+ np.exp((1.- arMiss) / self.sigma_cld)),
                            srate)



'''    #-------------------------------------------------------------------------
    #
    # The observation operator for hourly grid-cell predictions
    # Use MODIS history and hourly detection limits for whatever grid (preferably, same
    # as the one of tsM)
    # The FRP tsMatrix needs to include solar zenith angle
    #
    def observe_grid_cells_TSM(self, tsmFRP_mdl):
        #
        # reserve space for output
        DL = np.zeros(shape=(len(tsmFRP_mdl.times),len(tsmFRP_mdl.stations)))
        #
        # For times outside the MODIS period, use generic function
        idxOutside = np.logical_or(tsmFRP_mdl.times < self.tStart, tsmFRP_mdl.times > self.tEnd)
        idxInside = np.logical_not(idxOutside)
        if np.any(idxOutside):
            DL[idxOutside,:] = self.observe_grid_cells_TSM_gen(tsmFRP_mdl.vals[idxOutside,:])
        #
        # Times within the MODIS lifetime can be taken more accurately
        if np.any(idxInside):
            MM = daily_maps()  # meatdata from standard MODIS daily_maps
            # project points to grid
            fx, fy = self.MODIShist.grid.geo_to_grid(np.array(list(((s.lon for s in 
                                                                     tsmFRP_mdl.stations)))),
                                                     np.array(list(((s.lat for s in 
                                                                     tsmFRP_mdl.stations)))))
            ix = np.round(fx).astype(np.int32)
            iy = np.round(fy).astype(np.int32)
            idxt = np.where(idxInside)   # indices of times inside the MODIS period 
            for it in idxt:
                self.MODISreader_DL.seek(it)   # much faster than goto(time)
                mapDL = self.MODISreader_DL.read(1)
                self.MODISreader_noData.seek(it)   # much faster than goto(time)
                mapNoData = self.MODISreader_noData.read(1)
                DL[it,:] = mapDL(iy, ix) * (1.+ MM.cld_sigmoid_scale /
                                            (1.+ np.exp(-MM.cld_sigmoid_slope * 
                                                        (mapNoData(iy,ix) - MM.cld_sigmoid_shift))))
        #
        # DL has been found, do the final step
        #
        srate = np.nan   # not implemented yet
        return tsmFRP_mdl.vals / (1. + np.exp(-srate * (tsmFRP_mdl.vals / DL - 1.)))

    #-------------------------------------------------------------------------
    #
    # The observation operator for hourly grid-cell predictions
    # There is no MODIS pixel size, have to use the FRP-weighted mean pixel size
    #
#    def observe_grid_cells_TSM_gen(self, tsM_FRP_mdl):
'''
        
        
    