'''

Observation Operator for Fire Radiative Poewr products of low-orbit satellites.
Currently available coefficients are for MODIS and VIIRS
Supports both pixel- and grid-wise representation of the fire data.

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

@author: Mikhail Sofiev, Finnish Meteorological Institute
'''

import numpy as np, copy
from toolbox import silamfile, gridtools, supplementary as spp
import granule_MODIS as MxD_gran

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
    # The operator follows the notations in the paper Sofiev, 2025.
    # 
    def __init__(self, chInstrument, log):
        #
        # Instruments known this-far: MODIS, VIIRS
        #
        self.chInstr = {'MOD':'MODIS','MYD':'MODIS','MxD':'MODIS','MODIS':'MODIS',
                        'VNP':'VIIRS','VJ1':'VIIRS','Vxx':'VIIRS','VIIRS':'VIIRS'}[chInstrument]
        self.log = log
        # sigmoid rate 
        # day
        self.day_r_0 = {'MODIS': 0.07, 'VIIRS': 0.61}[self.chInstr]   # shift
        self.day_r_1 = {'MODIS': 1.26, 'VIIRS': 0.21}[self.chInstr]   # scale
        self.day_A_0 = {'MODIS': 0.,   'VIIRS': 0}[self.chInstr]      # shift
        # night
        self.night_r_0 = {'MODIS': 0.11, 'VIIRS': 2.12}[self.chInstr] # shift
        self.night_r_1 = {'MODIS': 1.06, 'VIIRS': 1.02}[self.chInstr] # scale
        self.night_A_0 = {'MODIS': 0.,   'VIIRS': 0.}[self.chInstr]   # shift
        # cut-off
        self.c_o = {'MODIS': 0.045, 'VIIRS': 0.05}[self.chInstr]      # same for day and night
        # detection limit
        # day
        self.DL_day_slope =   {'MODIS': 4.44,  'VIIRS': 6.17}[self.chInstr]
        self.DL_day_intercept ={'MODIS':0.52, 'VIIRS': 1.44}[self.chInstr]
        # night
        self.DL_night_slope =   {'MODIS': 4.43, 'VIIRS': 1.38}[self.chInstr]
        self.DL_night_intercept ={'MODIS': 1.01,  'VIIRS': 0.35}[self.chInstr]
        # Mean pixel size, FRP-weighted
        self.mean_pixel_size = {'MODIS': 2.45, 'VIIRS': 0.3}[self.chInstr]   # km2
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
        self.DL_mean = self.detection_limit(np.ones(shape=(2)) * self.mean_pixel_size,
                                            QA_cloud=11, sunglint=1, land=1, 
                                            ifNight=np.array([True,False]))
        self.srate_mean = self.slope_rate(np.ones(shape=(2)) * self.mean_pixel_size, 
                                          np.array([True,False]))


    #-----------------------------------------------------------------------
    # CORE functions
    #-----------------------------------------------------------------------
    #
    # The core and the final step of the observation operator
    # Observed FRP from modelled FRP using the detection limit and rate of the sigmoid
    # Note that if sigmoid < 1, it is not the FRP that should be scaled: we should assume 
    # lower probability to get this fire, but if you get it, it will be the input FRP.
    # This is the only way not to disturb the distribution.
    #
    def observe(self, arFRP_mdl, arDL, arRate):
        import warnings
        warnings.filterwarnings("error")
        try:
            obs = arRate * 0.0            # zeroes when arFRP_mdl << arDl
            prob = arRate * 0.0
            idxLrg = -arRate * (arFRP_mdl - arDL) < -20.0   # arFRP_mdl >> arDL
            obs[idxLrg] = arFRP_mdl[idxLrg]    # arFRP_mdl when arFRP_mdl >> arDL
            idxMid = np.logical_and(np.logical_not(idxLrg), -arRate * (arFRP_mdl - arDL) < 20.0)
            prob[idxMid] = np.maximum(0., 1. / (1. + np.exp(-arRate[idxMid] * (arFRP_mdl - arDL)[idxMid])) - 
                                         self.c_o) / (1. - self.c_o)
            rnd = np.random.random_sample(prob.size)  # as many as needed, uniform [0,1)
            idxNonZero = np.logical_and(idxMid, rnd < prob)
            obs[idxNonZero] = arFRP_mdl[idxNonZero]

        except:
            print('Got warning')
        warnings.resetwarnings()
        return obs


    #-----------------------------------------------------------------------
    #
    # Core of the detection limit, works for any array
    #
    def detection_limit(self, pixel_size, QA_cloud, sunglint, land, ifNight):
        #
        # Quality flags:
        # ('cld_mask','i1','Cloud mask(1=made)', '', self.BitFields.ifCloudAnalysed),
        # ('cld_clr', 'i1', 'cloud-clear: 0=cloud,1=uncertain,10=clear?,11=clear','',self.BitFields.QA),
        # ('d_n','i1', '1=day, 0=night','',self.BitFields.day_night), 
        # ('sun_gl','i1','0=sunglint','',self.BitFields.sunglint),
        # ('snow','i1','0=snow','',self.BitFields.snow),
        # ('water_land','i1','0=water,1=coast,10=desert,11=land','',self.BitFields.land)]:
        #
        DL = self.DL_day_slope * pixel_size + self.DL_day_intercept
        DL[ifNight] = self.DL_night_slope * pixel_size[ifNight] + self.DL_night_intercept
        
        # Clouds etc...
#        DL[QA_cloud < 11] *= 2   # something random, I have no clue on the actual factor...
#        DL[QA_cloud < 10] *= 10   # something random, I have no clue on the actual factor...
        DL[(QA_cloud < 1) | (sunglint == 0) | (land == 0)] = 1e6   # something big
        return DL.astype(np.float32)

    
    #-----------------------------------------------------------------------

    def slope_rate(self, pixel_size, ifNight):
        #
        # the slope rate
        #
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
    def observe_pixels(self, arFRP_mdl, pixel_size, QA_cloud, sunglint, land, ifNight):
        #
        # A basic function saying what MODIS/VIIRS/... would see if the actual value is as given
        # Nothing fancy here: detection limit is explicitly given
        # Assumption is: FRP << DL ==> FRP_obs = 0, 
        #                FRP >> DL ==> FRP_obs = FRP_mdl
        # transition is via sigmoid centred at DL and transition rate obtained from
        # fitting historical FRP observations
        #
        return self.observe(arFRP_mdl,
                            self.detection_limit(pixel_size, QA_cloud, sunglint, land, ifNight),
                            self.slope_rate(pixel_size, ifNight))


    #-----------------------------------------------------------------------

    def detection_limit_granule(self, granule):
        #
        # Detection limit for a single granule. This one is granule-type agnostic
        # but it requires QA fields
        #
        return self.detection_limit(np.repeat(granule.area[None,:],
                                              granule.BitFields.day_night.shape[0], axis=0),
                                    granule.BitFields.QA, 
                                    granule.BitFields.sunglint, 
                                    granule.BitFields.land, 
                                    granule.BitFields.day_night == 0)   #  0 = Night / 1 = Day

    #------------------------------------------------------------------------

    def slope_rate_granule(self, granule):
        #
        # and the slope rate
        #
        return self.slope_rate(np.repeat(granule.area[None,:],2030, axis=0),
                               granule.BitFields.day_night == 0)        #  0 = Night / 1 = Day


    #------------------------------------------------------------------------

    def observe_by_granule(self, inFRP, inCoord_line, inCoord_sample, gran):
        #
        # Observe a set of input FRP "by" the granule:
        # - the fires are inside the granule
        # - the line and sample coordinates are given, as well as the input FRP 
        # The output is the FRP as the particular satellite would retrive then should they fall in this granule 
        #
        return self.observe_pixels(inFRP, 
                                   gran.dS[inCoord_sample] * gran.dT[inCoord_sample],
                                   gran.BitFields.QA[inCoord_line, inCoord_sample],
                                   gran.BitFields.sunglint[inCoord_line, inCoord_sample],
                                   gran.BitFields.land[inCoord_line, inCoord_sample],
                                   gran.BitFields.day_night[inCoord_line, inCoord_sample] == 0)   #  0 = Night / 1 = Day
#        return FRP_obs
#        granOut = copy.deepcopy(gran)
#        granOut.FP = gran.FP.subset_fires(FRP_obs > 0)  # cut out zero-FRP fires
#        granOut.FP.FRP = FRP_obs[FRP_obs > 0]           # Assing the observed values
#        granOut.nFires = granOut.FP.nFires  # Do not forget to copy: if zero fire, this nFires will preclude FP analysis
#        return granOut

