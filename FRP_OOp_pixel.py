'''

Observation Operator for Fire Radiative Poewr products of low-orbit satellites.
Currently available coefficients are for MODIS and VIIRS
Supports both pixel- and grid-wise representation of the fire data.

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

@author: Mikhail Sofiev, Finnish Meteorological Institute
'''

import numpy as np
from toolbox import silamfile, gridtools, supplementary as spp
import MODIS_granule as MxD_gran

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

class FRP_observation_operator_pixel():
    
    #------------------------------------------------------------------------
    #
    # Use it to hard-code the parameters of the ObsOp
    # There are two parameters dependent on pixel size: detection limit and sigmoid rate
    # Rest is global constants like 0.0 and 1.0. They are hard-coded.
    # The operator follows the notations in the paper Sofiev, 2022.
    # 
    def __init__(self, chInstrument, log):
        #
        # Instruments known this-far: MODIS, VIIRS
        #
        self.chInstr = {'MOD':'MODIS','MYD':'MODIS','MODIS':'MODIS',
                        'VNP14IMG':'VIIRS','VJ114IMG':'VIIRS','VIIRS':'VIIRS'}[chInstrument]
        self.log = log
        # sigmoid rate 
        # day
        self.day_r_0 = {'MODIS':0.106866, 'VIIRS':0.662415}[self.chInstr]   # shift
        self.day_r_1 = {'MODIS':1.17774, 'VIIRS':0.215982}[self.chInstr]    # scale
        self.day_A_0 = {'MODIS':0., 'VIIRS':0}[self.chInstr]            # shift
        # night
        self.night_r_0 = {'MODIS':0., 'VIIRS':3.41036}[self.chInstr]   # shift
        self.night_r_1 = {'MODIS':2.22502, 'VIIRS':0.327912}[self.chInstr] # scale
        self.night_A_0 = {'MODIS':0.139678, 'VIIRS':0.188783}[self.chInstr] # shift
        # detection limit
        # day
        self.DL_day_slope = {'MODIS':4.4817, 'VIIRS':6.16297}[self.chInstr]
        self.DL_day_intercept = {'MODIS':1.21797, 'VIIRS':1.49392}[self.chInstr]
        # night
        self.DL_night_slope = {'MODIS':4.26839, 'VIIRS':1.30424}[self.chInstr]
        self.DL_night_intercept = {'MODIS':-0.530978, 'VIIRS':0.187262}[self.chInstr]
        # Mean pixel size, FRP-weighted
        self.mean_pixel_size = {'MODIS':2.45, 'VIIRS':0.3}[self.chInstr]   # km2
#        #
#        # For the period of MODIS observations, we can have exact pattern, need history
#        #
#        if chMODIShist_FNmTempl is None:
#            self.MODIShist = None
#        else:
#            self.MODIShist = silamfile.SilamNCFile(chMODIShist_FNmTempl)
#            self.MODISreader_DL = self.MODIShist.get_reader('detection_limit')
#            self.MODISreader_noData = self.MODIShist.get_reader('missing_cells')
#            self.tStart = self.MODIShist.t()[0]
#            self.tEnd = self.MODIShist.t()[-1]
        #
        # For the period outside the satellite observations or if we do not need high precision
        # make a "typical" clear-sky detection limit and slope
        #
        self.DL_mean = self.detection_limit(np.ones(shape=(2)) * self.mean_pixel_size, np.array([True,False]))
        self.srate_mean = self.slope_rate(np.ones(shape=(2)) * self.mean_pixel_size, np.array([True,False]))


    #-----------------------------------------------------------------------
    # CORE functions
    #-----------------------------------------------------------------------
    #
    # The core and the final step of the observation operator
    # Observed FRP from modelled FRP using the detection limit and rate of the signoid
    #
    def observe(self, arFRP_mdl, arDL, arRate):
        return arFRP_mdl / (1. + np.exp(-arRate * (arFRP_mdl - arDL)))


    #-----------------------------------------------------------------------
    #
    # Core of the detection limit, works for any array
    #
    def detection_limit(self, pixel_size, ifNight):
        DL = self.DL_day_slope * pixel_size + self.DL_day_intercept
        DL[ifNight] = self.DL_night_slope * pixel_size[ifNight] + self.DL_night_intercept
        return DL.astype(np.float32)
    #
    # and the slope rate
    #
    def slope_rate(self, pixel_size, ifNight):
        srate = self.day_r_0 + self.day_r_1 / (pixel_size - self.day_A_0)
        srate[ifNight] = self.night_r_0 + self.night_r_1 / (pixel_size[ifNight] - self.night_A_0)
        return srate.astype(np.float32)


    #-----------------------------------------------------------------------
    # Input-output-specific functions
    #-----------------------------------------------------------------------
    #
    # Single-granule operator: hourly model prediction for individual points. 
    # Colocated with the satellite: need pixel size and solar zenit angle (rather, the day-night switch)
    #
    def observe_pixels(self, arFRP_mdl, pixel_size, ifNight):
        #
        # A basic function saying what MODIS/VIIRS/... would see if the actual value is as given
        # Nothing fancy here: detection limit is explicitly given
        # Assumption is: FRP << DL ==> FRP_obs = 0, 
        #                FRP >> DL ==> FRP_obs = FRP_mdl
        # transition is via sigmoid centred at DL and transition rate obtained from
        # fitting historical FRP observations
        #
        return self.observe(arFRP_mdl,
                            self.detection_limit(pixel_size, ifNight),
                            self.slope_rate(pixel_size, ifNight))


    #-----------------------------------------------------------------------
    #
    # A shortcut: detection limit for a single granule. This one is granule-type agnostic
    #
    def detection_limit_granule(self, granule):
        return self.detection_limit(np.repeat(granule.area[None,:],
                                              granule.BitFields.day_night.shape[0], axis=0),
                                    granule.BitFields.day_night == 1)
    #
    # and the slope rate
    #
    def slope_rate_granule(self, granule):
        return self.slope_rate(np.repeat(granule.area[None,:],2030, axis=0),
                               granule.BitFields.day_night == 1)
    #
    # And observe granule assuming that it is mutable, i.e. sent here as a pointer, to which we can add
    #
    def observe_granule(self, gran):
        gran.FP_frp_obs = self.observe_pixels(gran.FP_frp, gran.dS*gran.dT, np.abs(gran.FP_SolZenAng) > 90.) 
        return gran

