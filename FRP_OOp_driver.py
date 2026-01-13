'''

Driver for the FRP observation operator
The top-most module for the FRP OOp set

Created on 14.1.2022

@author: sofievm
'''

import os, sys, numpy as np, datetime as dt, glob, pickle, copy
import matplotlib as mpl
from mpl_toolkits import basemap
from support import suntime
from toolbox import supplementary as spp, gridtools, silamfile
import FRP_OOp_grid, FRP_OOp_pixel, process_satellites as sat_proc
import granule_MODIS, granule_VIIRS
import land_use as LU_module
import timeit
import fire_records

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
if os.getenv("DISABLE_MPI","FALSE") == "TRUE":   # FALSE is default, i.e. MPI will be active unless disabled
    print("mpi disabled, force single-process operation")
    mpisize_loc = 1
    mpirank_loc = 0
    chMPI = ''
    comm = None
else:
    try:
        # slurm loader environment
        mpirank_loc = int(os.getenv("SLURM_PROCID",None))
        mpisize_loc = int(os.getenv("SLURM_NTASKS",None))
        chMPI = '_mpi%03g' % mpirank_loc
        comm = None
        print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize_loc)
    except:
        # not in puhti - try usual way
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            mpisize_loc = comm.size
            mpirank_loc = comm.Get_rank()
            chMPI = '_mpi%03g' % mpirank_loc
            print('MPI operation, mpisize= %i, %s' % (mpisize_loc, chMPI))
        except:
            print("mpi4py failed, single-process operation")
            mpisize_loc = 1
            mpirank_loc = 0
            chMPI = ''
            comm = None

#
# The top function: observe
#
def observe_FRP_gridded(mdlFRP, grid, timeUTC, chInstrument, chSatFNmTempl, chDirWrk, ifNight_=None):
    #
    # Get solar zenith angle if it is not given. Note that it might be an array, i.e. mutable
    #
    if ifNight_ is None:
        with suntime.Sun(grid.lons, grid.lats) as sun:
            ifNight = np.logical_or(timeUTC < sun.get_sunrise_time(timeUTC),
                                    timeUTC > sun.get_sunset_time(timeUTC))
    else:
        ifNight = ifNight_
    #
    # Create an object for the observation operator and use it to observe
    # We observe the whole map at once
    # 
    OOp = FRP_OOp_grid.FRP_observation_operator_grid(chInstrument, chSatFNmTempl, 
                                                     spp.log.log(os.path.join(chDirWrk,'OOp.log')))
    

#######################################################################################################

def VIIRS_2_MODIS_grid(OOp, grid, gridName, fRecs_VIIRS, fRecs_MODIS, year, dirOut, log):
    #
    # Project them to a predefined grid, which should resemble MODIS resolution
    #
    # The key missing pieces:
    # VIIRS flies as a couple, most-probably recording each fire twice at each 
    # dual-craft overpass. Averaging is a good idea: they register the same
    # fire event, as seen from the point of view of the gridded analysis
    # So:
    # - project the fires to the cells of the OOp grid
    # - sum-up accounting for diurnal variation inside each cell 
    #
    fX, fY = grid.geo_to_grid(fRecs_VIIRS.lon, fRecs_VIIRS.lat)  # for all times
    ix = np.round(fX).astype(np.int64)
    iy = np.round(fY).astype(np.int64)
    i1dLst = ix + iy * grid.nx
    iHrLst = np.round(fRecs_VIIRS.time / 3600).astype(np.int32) + fRecs_VIIRS.timeStart.hour
    iFireAsMODIS = 0
    # VIIRS_2_MODIS storage place
    FRP_asMODIS = np.zeros(shape=(100000),dtype=np.float32)
    hr_asMODIS = np.zeros(shape=(100000),dtype=np.int32)
    i1d_asMODIS = np.zeros(shape=(100000),dtype=np.int32)
    sza_asMODIS = np.zeros(shape=(100000),dtype=np.float32)
    #
    # Processing goes hour by hour - for those that exist in the records
    #
    iHr_uniq = np.array(sorted(list(set(iHrLst))))

    for iHr in iHr_uniq:
        
        if iHr > 50: break
        #
        # Sum-up the VIIR pixels that came to the same MODIS pixel
        #
        maskHr = iHrLst == iHr
        i1d_uniq = np.array(list(set(i1dLst[maskHr])))  # for this hour
        FRP_uniq = np.zeros(shape=(i1d_uniq.size),dtype=np.float32)  # place holder for FRP, this hour, output grid
        sza_uniq = np.zeros(shape=(i1d_uniq.size),dtype=np.float32)  # place holder for Solar Zenith angle, this hour, output grid
        dxs_uniq = np.zeros(shape=(i1d_uniq.size),dtype=np.float32)  # place holder for dx, this hour, output grid
        dys_uniq = np.zeros(shape=(i1d_uniq.size),dtype=np.float32)  # place holder for dy, this hour, output grid
        FRP_hr = fRecs_VIIRS.FRP[maskHr]            # FRP to aggregate to the output grid
        SZA_hr = fRecs_VIIRS.SolZenAng[maskHr]
        i1dLst_hr = i1dLst[maskHr]          # unique 1d grid indices of this hour
        if iFireAsMODIS + i1dLst_hr.size > FRP_asMODIS.size:
            FRP_asMODIS.resize(FRP_asMODIS.size * 2)
            hr_asMODIS.resize(FRP_asMODIS.size * 2)
            i1d_asMODIS.resize(FRP_asMODIS.size * 2)
            sza_asMODIS.resize(FRP_asMODIS.size * 2)

#                    ix_hr = np.mod(i1dLst_hr, grid_def[0].nx)       # 2d ix for this hour
#                    iy_hr = (i1dLst_hr - ix_hr) // grid_def[0].ny   # 2d iy for this hour
#                    lon_hr, lat_hr = grid_def[0].grid_to_geo(ix_hr, iy_hr)  # corresponding lons and lats
#                    dxs_hr, dys_hr = grid_def[0].cell_dimensions(lon_hr, lat_hr)
        #
        # Finally, FRP for this hour
        #
        for ii, i1d in enumerate(i1d_uniq):
            mask_i1d = i1dLst_hr == i1d
#                        if np.mod(ii, 100) == 0: print(ii, i1d)
            FRP_uniq[ii] = np.sum(FRP_hr[mask_i1d])
            sza_uniq[ii] = SZA_hr[mask_i1d][0]  # single cell, single hour...
            ix_uniq = np.mod(i1d, grid.nx)       # 2d ix for this hour
            iy_uniq = (i1d - ix_uniq) // grid.ny   # 2d iy for this hour
            lon, lat = grid.grid_to_geo(ix_uniq, iy_uniq)
            dxs_uniq[ii], dys_uniq[ii] = grid.cell_dimensions(lon, lat)
            # prepare cell sizes and zenith angle for these pixels 
        
        FRP_asMODIS[iFireAsMODIS:iFireAsMODIS+i1d_uniq.size] = OOp.observe_pixels(
                               FRP_uniq,           # FRP of VIIRS summed to MODIS grid
                               dxs_uniq * dys_uniq * 1e-6,   # pixel size of MODIS grid
                               sza_uniq >= 90)       # day/night?

        hr_asMODIS[iFireAsMODIS:iFireAsMODIS+i1d_uniq.size] = iHr
        i1d_asMODIS[iFireAsMODIS:iFireAsMODIS+i1d_uniq.size] = i1d_uniq
        sza_asMODIS[iFireAsMODIS:iFireAsMODIS+i1d_uniq.size] = sza_uniq
        
        log.log('Hour: %i, FRP VIIRS: %g, FRP as MODIS: %g, nFire cells: %i' % 
                (iHr, np.sum(FRP_uniq), np.sum(FRP_asMODIS[iFireAsMODIS:iFireAsMODIS+i1d_uniq.size]), i1d_uniq.size))
        iFireAsMODIS += i1d_uniq.size
        
#                print('year %i, hour %i, grid %s, VIIRS nFires = %i, nUnique_idx = %i, ratio: %g' %
#                      (year, iHr, grid_def[1], i1dLst.shape[0], i1d_uniq.shape[0], i1dLst.shape[0] / i1d_uniq.shape[0]))
#                #
#                # Cell sizes for VIIRS fires
#                #
#                dxs, dys = grid_def[0].cell_dimensions(fRecs[year].lon[:fRecs[year].nFires],
#                                                       fRecs[year].lat[:fRecs[year].nFires])  # the whole grid
#                #
#                # Apply pixel-wise OOp
#                #
#                FRP_MODIS = OOp.observe_pixels(fRecs[year].FRP[:fRecs[year].nFires], 
#                                               dxs * dys * 1e-6,         # pixel size
#                                               fRecs[year].SolZenAng[:fRecs[year].nFires] >= 90)
#                t2 = dt.datetime.now()
#                print('Done within: ',t2-t1)

    log.log('year: %i, grid: %s' % (year, gridName))
    log.log('FRP_VIIRS, TW = %g, VNP/VJ1 %g %g' % 
            (np.sum(fRecs_VIIRS.FRP) * 1e-6,
             np.sum(fRecs_VIIRS.FRP[fRecs_VIIRS.satellite == b'N']) * 1e-6,
             np.sum(fRecs_VIIRS.FRP[fRecs_VIIRS.satellite == b'1']) * 1e-6))
    log.log('FRP_as_MODIS,TW = %g' % (np.sum(FRP_asMODIS) * 1e-6))
    log.log('FRP_MODIS_true, TW = %g, MOD/MYD %g %g \n\n' % 
            (np.sum(fRecs_MODIS.FRP) * 1e-6,
             np.sum(fRecs_MODIS.FRP[fRecs_MODIS.satellite == b'T']),
             np.sum(fRecs_MODIS.FRP[fRecs_MODIS.satellite == b'A'])))
    #
    # Make the fire_records object and store it 
    # 
    fRec_MODIS_from_VIIRS = sat_proc.fire_records(log)
    
    ix = np.mod(i1d_asMODIS, grid.nx)       # 2d ix
    iy = (i1d_asMODIS - ix) // grid.ny   # 2d iy
    lon, lat = grid.grid_to_geo(ix, iy)
    dxs, dys = grid.cell_dimensions(lon, lat)
    arZero = FRP_asMODIS[:iFireAsMODIS] * 0.0
    times = hr_asMODIS.astype(np.int64) * 3600 # to seconds since the timeStart
    
    # Fire records from dictionary
    fRec_MODIS_from_VIIRS.from_dic({'lon': lon[:iFireAsMODIS],
                                    'lat': lat[:iFireAsMODIS],
                                    'time': times[:iFireAsMODIS],
                                    'FRP': FRP_asMODIS[:iFireAsMODIS],
                                    'dS': dxs[:iFireAsMODIS] /1000.,
                                    'dT': arZero + dys /1000.,
                                    'sza': sza_asMODIS[:iFireAsMODIS],
                                    'ViewZenAng': arZero, 'T4': arZero, 'T4b':  arZero,
                                    'T11':  arZero, 'T11b': arZero, 'TA': arZero,
                                    'ix': ix[:iFireAsMODIS],
                                    'iy': iy[:iFireAsMODIS],
                                    'i_line': arZero,'i_sample':  arZero, 'LU': arZero,
                                    'satellite':'OOP_VIIRS_2_MODIS',
                                    'grid': grid,
                                    'gridName':gridName,
                                    'timezone': fRecs_VIIRS.timezone,
                                    'timeStart': fRecs_VIIRS.timeStart,
                                    'QA_flag': fRecs_VIIRS.QA_flag,
                                    'QA_msg': fRecs_VIIRS.QA_msg})

    fRec_MODIS_from_VIIRS.to_nc(os.path.join(dirOut,'MODIS_from_VIIRS_%s_%i.nc4' % (gridName, year)))


#######################################################################################################
        
def VIIRS_2_MODIS_granules(fRecs_VIIRS_day, chFNmMODIS_14_templ, chFNmMODIS_35_templ, 
                           chFNmVIIRS_14_templ, chFNmVIIRS_03_templ, 
                           today, ifDrawGranules, dirOut, log, timesDone=None):
    #
    # Pixel-wise OOp application
    # - get one day of VIIRS fires
    # - scan this day MODIS granules one by one projecting fires on them and checking
    #   if MODIS would be able to see them
    # - compare with what MODIS actually saw for the specific granule
    #
    # Symmetrization remark:
    # Projecting-and-observing VIIRS on MODIS granule means that MODIS retrievals go as-is
    # while they may need to be as VIIRS would have seen them. Then we would compare
    # VIIRS as would be seen by MODIS with MODIS as would be seen by VIIRS, i.e. the problem
    # becomes symmetrical. Since VIIRS is more sensitive, the only non-trivial part of 
    # MODIS_as_VIIRS vision is clouds of VIIRS: due to different overpass times they will differ
    #
    if ifDrawGranules: 
        spp.ensure_directory_MPI(os.path.join(dirOut,'pics','OOp_MYD14_granules'))
#        spp.ensure_directory_MPI(os.path.join(dirOut,'pics','MYD14_granules'))
    #
    # Cycle over MODIS granules
    #
    totFRP_MODIS = []
    totFRP_MODIS_as_VIIRS = []
    totFRP_VIIRS_in_MODISgran = []
    totFRP_VIIRS_as_MODIS = []
    gran_time = []
    VIIRS_grans = {}
    #
    # Observation operators
    #
    OOp_MODIS = FRP_OOp_pixel.FRP_observation_operator_pixel('MODIS',log)
    OOp_VIIRS = FRP_OOp_pixel.FRP_observation_operator_pixel('VIIRS',log)    #
    #
    # Process one day, granule by granule
    #
    now = today
    while now < today + spp.one_day:
        
        if timesDone is not None:
            if timesDone[min(np.searchsorted(timesDone,now), len(timesDone)-1)] == now: # done before
                print('Done already:', now)
                now += spp.one_minute * 5
                continue
        
        chFNmFRP = glob.glob(now.strftime(chFNmMODIS_14_templ))
        chFNmGEO = glob.glob(now.strftime(chFNmMODIS_35_templ))
        if len(chFNmFRP) == 1 and len(chFNmGEO) == 1:
            log.log(now.strftime('%Y%m%d-%H%M:') + ' Granule: ' + chFNmFRP[0] + ',  ' + chFNmGEO[0])
        else:
            log.log(now.strftime('%Y%m%d-%H%M:') + 'No such granule or more than one')
            now += spp.one_minute * 5
            continue
        #
        # get the data
        #
        gran = granule_MODIS.granule_MODIS(now_UTC = None,    
                                           chFRPfilesTempl = chFNmFRP[0], 
                                           chAuxilFilesTempl = chFNmGEO[0],
                                           log = log)
        chFRP_Basename = os.path.split(chFNmFRP[0])[-1]
#        if chFRP_Basename == 'MYD14.A2020001.0100.061.2020321040157.hdf':
#            print('Here')

        if not gran.pick_granule_data_IS4FIRES_v3_0():
            log.log('Unreadable granule: %s, %s' % (now.strftime(chFNmMODIS_14_templ), 
                                                    now.strftime(chFNmMODIS_35_templ)))
            now += spp.one_minute * 5
            continue
        if ((np.nanmin(gran.lon) < -180) | (np.nanmax(gran.lon) > 180) | 
            (np.nanmin(gran.lat) < -90) | (np.nanmax(gran.lat) > 90) |
            np.any(np.isnan(gran.lon + gran.lat))):
            log.log('Weird granule: min/max of lon: %g/%g and/or lat: %g/%g, nans: %i, %s, %s ' % 
                    (np.nanmin(gran.lon),np.nanmax(gran.lon), np.nanmin(gran.lat), np.nanmax(gran.lat),
                     np.sum(np.isnan(gran.lon + gran.lat)),
                     now.strftime(chFNmMODIS_14_templ), now.strftime(chFNmMODIS_35_templ)))
            now += spp.one_minute * 5
            continue
        #
        # How would VIIRS, in its overpass, see these MODIS fires?
        # Need to check a few VIIRS granules to find the one that covers the MODIS fires
        # Within +- 2 hours.
        #
        timesTmp = np.array(list((dt.datetime(now.year,now.month,now.day)-spp.one_day + 
                                  spp.one_minute * 6 * i for i in range(720))))
        idxTimesNearby = (timesTmp >= now - spp.one_hour * 2) & (timesTmp <= now + spp.one_hour * 2)  # AQUA is mostly after SNP/NOAA
        # placeholder for MODIS_as_VIIRS FRP
        if gran.nFires > 0:
            print('Found %i MODIS fires, the first one at (%gE, %gN)' % (gran.nFires, gran.FP.lon[0], gran.FP.lat[0]))
            FRP_MODIS_as_VIIRS = np.zeros(shape=gran.FP.FRP.shape, dtype=np.float32) * np.nan
            # scan all granules around this time
            for time2try in timesTmp[idxTimesNearby]:
                chFNmVIIRSFRP = glob.glob(time2try.strftime(chFNmVIIRS_14_templ))
                chFNmVIIRSGEO = glob.glob(time2try.strftime(chFNmVIIRS_03_templ))
                if len(chFNmVIIRSFRP) == 1 and len(chFNmVIIRSGEO) == 1:
                    pass #log.log(time2try.strftime('%Y%m%d-%H%M:') + ' VIIRS granule: ' + chFNmVIIRSFRP[0] + ',  ' + chFNmVIIRSGEO[0])
                else:
                    log.log(time2try.strftime('%Y%m%d-%H%M:') + 'No such granule or more than one')
                    continue
                # Seen that granule before?
                if chFNmVIIRSFRP[0] in VIIRS_grans:
                    # check coverage
                    minlon, maxlon, minlat, maxlat = VIIRS_grans[chFNmVIIRSFRP[0]]
                    if not (np.any(minlon < gran.FP.lon) & np.any(maxlon > gran.FP.lon) &
                            np.any(minlat < gran.FP.lat) & np.any(maxlat > gran.FP.lat)):
                        continue
                granVIIRS = granule_VIIRS.granule_VIIRS(now_UTC = None,    
                                                            chFRPfilesTempl = chFNmVIIRSFRP[0], 
                                                            chAuxilFilesTempl = chFNmVIIRSGEO[0],
                                                            log = log)
                if not granVIIRS.pick_granule_data_IS4FIRES_v3_0():
                        log.log('Unreadable VIIRS granule: %s, %s' % (time2try.strftime(chFNmVIIRS_14_templ), 
                                                                      time2try.strftime(chFNmVIIRS_03_templ)))
                        continue
                minlon = np.min(granVIIRS.lon)
                maxlon = np.max(granVIIRS.lon)
                minlat = np.min(granVIIRS.lat)
                maxlat = np.max(granVIIRS.lat)
                VIIRS_grans[chFNmVIIRSFRP[0]] = (minlon, maxlon, minlat, maxlat)

                nInside = np.sum((minlon < gran.FP.lon) & (maxlon > gran.FP.lon) & 
                                 (minlat < gran.FP.lat) & (maxlat > gran.FP.lat))
                print('Newly red VIIRS granule %s covers (%g-%gE, %g-%gN), %i fires inside envelope' % 
                      (os.path.split(chFNmVIIRSFRP[0])[-1], minlon, maxlon, minlat, maxlat, nInside))
                if (nInside == 0): continue
                #
                # The flag is 0 for points far from the granule, 1 inside the granule envelope, 2 inside the granule
                #
                flagMODISinVIIRSnear, idxMODISinVIIRScoords = granVIIRS.project_points_to_granule(
                                                                            gran.FP.lon, gran.FP.lat)
                if flagMODISinVIIRSnear is None: continue  # not this VIIRS granule
                else: 
                    if not np.any(flagMODISinVIIRSnear == 2): continue  # not this VIIRS granule, nothing inside
                #
                # Update those MODIS FRPs that fell into this VIIRS granule
                #
                FRP_MODIS_as_VIIRS[flagMODISinVIIRSnear == 2] = OOp_VIIRS.observe_by_granule(
                                            gran.FP.FRP[flagMODISinVIIRSnear == 2],
                                            idxMODISinVIIRScoords[0,:][idxMODISinVIIRScoords[0,:] >=0],
                                            idxMODISinVIIRScoords[1,:][idxMODISinVIIRScoords[1,:] >=0],
                                            granVIIRS)
                    
#                for fMasVTmp, fModTmp in zip(FRP_MODIS_as_VIIRS[flagMODISinVIIRSnear == 2],
#                                             gran.FP.FRP[flagMODISinVIIRSnear == 2]):
#                    print('Observed by VIIRS OOp: FRP from %g to %g' % (fModTmp, fMasVTmp))

                # anything left to search for?
                if np.all(np.isfinite(FRP_MODIS_as_VIIRS)): 
                    break
                else:
                    print('Filled %i fires, remained %i' % (np.sum(np.isfinite(FRP_MODIS_as_VIIRS)), 
                                                            np.sum(np.isnan(FRP_MODIS_as_VIIRS)))) 
            
            if np.any(np.isnan(FRP_MODIS_as_VIIRS)): 
                log.log('Failed to find VIIRS granule for some MODIS fires for ' + chFRP_Basename + 
                        ', skip the MODIS granule')
                now += spp.one_minute * 5
                continue
            #
            # Store the MODIS-as-VIIRS fires
            #
            granMODIS_as_VIIRS = copy.deepcopy(gran)
            granMODIS_as_VIIRS.FP.FRP[:] = FRP_MODIS_as_VIIRS[:]
        else:
            # no MODIS fires, nothing to look at from VIIRS standpoint
            FRP_MODIS_as_VIIRS = np.array([], dtype=np.float32)
            granMODIS_as_VIIRS = copy.deepcopy(gran)   # still copy: the FP will be overwritten
        #
        # Project fires from VIIRS to this granule
        # Uses heavily modified optimization routine from SILAM anygrid
        # The routine works in Cartesian coordinates
        # returns indices of fires that have been processed (close to the granule or in it)
        #
        # We require close-time fires.
        #
        fRecs_VIIRS_day_4gran = fRecs_VIIRS_day.subset_time(gran.now_UTC - (spp.one_hour*2), 
                                                            gran.now_UTC + (spp.one_hour*2))
        
        flagVIIRSinMODISnear, idxCoords = gran.project_points_to_granule(fRecs_VIIRS_day_4gran.lon, 
                                                                         fRecs_VIIRS_day_4gran.lat)
        #
        # Anything worth of processing?
        #
        if idxCoords is None:  # no fires were tried to the granule
#            print(now.strftime('None returned: No VIIRS fires in MODIS granule %Y%m%d-%H%M'))
            now += spp.one_minute * 5 
            continue
        if np.all(idxCoords < 0):  # fires were tried but appeared not in granule
#            print(now.strftime('No VIIRS fires in MODIS granule %Y%m%d-%H%M'))
            now += spp.one_minute * 5 
            continue
        print('%i daily VIIRS fires, %i close in time, %i near, %i in MODIS granule' % 
              (fRecs_VIIRS_day.nFires, fRecs_VIIRS_day_4gran.nFires, 
               np.sum(flagVIIRSinMODISnear == 1), np.sum(flagVIIRSinMODISnear == 2)))
        #
        # Draw the original granule ?
        #
#        if ifDrawGranules:
#            gran.draw_granule(os.path.join(dirOut,'pics', 'MYD14_granules', chFRP_Basename + '.png'))
        #
        # Preserve the original MODIS granule and reserve space for VIIRS-based granule
        #
        granMODIS = copy.deepcopy(gran)            # original MODIS
        granVIIRS_as_MODIS = copy.deepcopy(gran)   # VIIRS observations through MODIS OOp
        #
        # Create the MODIS granule with VIIRS fires and store for future use 
        #
        # Pick VIIRS fires nearby
        #
        frVIIRS_inGran = fRecs_VIIRS_day_4gran.subset_fires(flagVIIRSinMODISnear == 2)  # VIIRS fires that are near the MODIS granule
        #
        # Store the VIIRS fires into this granule - for comparison
        #
        gran.FP = frVIIRS_inGran
        # 
        # Create VIIRS-as-MODIS time series
        # Apply MODIS OOp and create the MODIS granule with VIIRS fires after OOp applied
        #
        FRP_VIIRS_as_MODIS_obsAr = OOp_MODIS.observe_by_granule(frVIIRS_inGran.FRP,               # FRP to observe 
                                                          idxCoords[0,:][idxCoords[0,:] >= 0],  # line coord in the granule # frVIIRS_inGran.line, 
                                                          idxCoords[1,:][idxCoords[0,:] >= 0],  # sample coord in the granule # frVIIRS_inGran.sample, 
                                                          gran)            # granule that observes the FRP
        #
        # Create new fire records with observed FRP and other parameters taken from the granule (those that are availale)
        #
        granVIIRS_as_MODIS.make_fire_records(FRP_VIIRS_as_MODIS_obsAr, 
                                             idxCoords[0,:][idxCoords[0,:] >= 0],
                                             idxCoords[1,:][idxCoords[0,:] >= 0])

        # MODIS granule with MODIS_as_VIIRS-processed fires
        spp.ensure_directory_MPI(os.path.join(dirOut, 'MODIS_with_MODIS_as_VIIRS_fires'))
        granMODIS_as_VIIRS.to_nc(os.path.join(dirOut,'MODIS_with_MODIS_as_VIIRS_fires',
                                              chFRP_Basename.replace('D14','D14_as_VIIRS').replace('.hdf','.nc4')))
        # intermediate: MODIS granule with VIIRS fires unprocessed
        spp.ensure_directory_MPI(os.path.join(dirOut, 'MODIS_with_VIIRS_fires'))
        gran.to_nc(os.path.join(dirOut,'MODIS_with_VIIRS_fires',
                                chFRP_Basename.replace('D14','D_VIIRS').replace('.hdf','.nc4')))
        
        # the final answer
        spp.ensure_directory_MPI(os.path.join(dirOut, 'MODIS_with_VIIRS_as_MODIS_fires'))
        granVIIRS_as_MODIS.to_nc(os.path.join(dirOut,'MODIS_with_VIIRS_as_MODIS_fires',
                                              chFRP_Basename.replace('D14','D_VIIRSasD14').replace('.hdf','.nc4')))
        # draw and compute totals
        
        if granMODIS.nFires == 0: 
            sumMODIS = 0
            sumMODIS_as_VIIRS = 0
        else: 
            sumMODIS = np.sum(granMODIS.FP.FRP[:granMODIS.nFires])
            sumMODIS_as_VIIRS = np.sum(granMODIS_as_VIIRS.FP.FRP[:granMODIS_as_VIIRS.nFires])
        
        log.log(gran.now_UTC.strftime('%Y%m%d-%H%M: ') + 'FRP for %s: MODIS=%g, MODIS_as_VIIRS=%g, VIIRS=%g, VIIRS_as_MODIS=%g\n' % 
                (chFRP_Basename, sumMODIS, sumMODIS_as_VIIRS, np.sum(gran.FP.FRP), np.sum(granVIIRS_as_MODIS.FP.FRP)))
        totFRP_MODIS.append(sumMODIS)
        totFRP_MODIS_as_VIIRS.append(sumMODIS_as_VIIRS)
        totFRP_VIIRS_in_MODISgran.append(np.sum(gran.FP.FRP))
        totFRP_VIIRS_as_MODIS.append(np.sum(granVIIRS_as_MODIS.FP.FRP))
        gran_time.append(gran.now_UTC)
        
        if ifDrawGranules:
            draw_four_grans(OOp, granMODIS, 'MODIS', gran, 'VIIRS', granVIIRS_as_MODIS, 'VIIRS_as_MODIS',
                            FRP_MODIS_as_VIIRS, 'MODIS_as_VIIRS',
                            os.path.join(dirOut,'pics','OOp_MYD14_granules',
                                         'OOp_' + chFRP_Basename + '.png'))
        #
        # Done with this granule, store the outcome
        #
        log.flush()

        now += spp.one_minute * 5
    #
    # Draw the collected granule totals
    #
    if len(totFRP_VIIRS_in_MODISgran) > 0:
        fig, ax = mpl.pyplot.subplots(1)
        p1 = ax.scatter(totFRP_MODIS, totFRP_VIIRS_in_MODISgran, label='VIIRS_in_MODISgran')
        p2 = ax.scatter(totFRP_MODIS, totFRP_VIIRS_as_MODIS, label='VIIRS_as_MODIS')
        ax.set_xlim(0,np.max(totFRP_VIIRS_in_MODISgran))
        ax.set_ylim(0,np.max(totFRP_VIIRS_in_MODISgran))
        ax.legend(loc=4)
        ax.set_xlabel('MODIS FRP, [MW]')
        ax.set_ylabel('VIIRS, VIIRS_as_MODIS FRP, [MW]')
        ax.legend()
        mpl.pyplot.savefig(os.path.join(dirOut,'pics',today.strftime('MODIS_vs_OOp_of_VIIRS_%Y%m%d.png')))
    


######################################################################################################

def draw_four_grans(OOp, granMODIS, nmMODIS, granVIIRS, nmVIIRS, granVIIRS_as_MODIS, nmVIIRS_as_MODIS, 
                    granMODIS_as_VIIRS, nmMODIS_as_VIIRS, chOutFNm):
    #
    # Draws three granules: original MODIS, MODIS with VIIRS fires, and MODIS with VIIRS fires after OOp
    #
        #
        # Area covered by this swath, with a bit free space around:
        #
        # This is to draw a few cross-sections along the scan, just 50 point
#        figTmp, ax = mpl.pyplot.subplots(1,1, figsize=(16,10))
#        for shift in range(0,self.lat_1km.shape[1],100):
#            pltLat = mpl.pyplot.plot(range(50), self.lat_1km[:50,shift], marker='.')
#        mpl.pyplot.show()
#        sys.exit()
        
        fig, axes = mpl.pyplot.subplots(1,4, figsize=(16,5))
        minLon = np.nanmin(granMODIS.lon)
        minLat = np.nanmin(granMODIS.lat)
        maxLon = np.nanmax(granMODIS.lon)
        maxLat = np.nanmax(granMODIS.lat)
        cmap = mpl.pyplot.get_cmap('cool')
        ixAx = 0
        chFires = ' no fires'
        # prepare input data: fires may not be present in some granules
        detLimRef = OOp.detection_limit_granule(granMODIS)
        dicFP = {}
        FRPmax = 0
        
        for nm, gran in [(nmMODIS,granMODIS), (nmVIIRS,granVIIRS), 
                         (nmVIIRS_as_MODIS, granVIIRS_as_MODIS), (nmMODIS_as_VIIRS, granMODIS_as_VIIRS)]:
            if gran.nFires > 0: 
                dicFP[nm] = (gran.FP.FRP, gran.FP.lon, gran.FP.lat)
                FRPmax = max(FRPmax, np.max(gran.FP.FRP))
            else: 
                dicFP[nm] = ([],None,None)
#        upLim = max(FRPmax, np.max(detLimRef))
        upLim = min(FRPmax, np.max(detLimRef))
        norm=mpl.pyplot.Normalize(0, 50)  # MODIS DL is up to ~40, clouds reach much higher 
        
        for var, chTxt, name, cm in [(detLimRef,'MODIS DL + FRP MODIS, [%g MW]', nmMODIS, 'cool'),
                                     (detLimRef,'MODIS DL + FRP MODIS_as_VIRS, [%g MW]', nmMODIS_as_VIIRS, 'cool'),
                                     (detLimRef,'MODIS DL + FRP VIIRS, [%g MW]', nmVIIRS, 'cool'),
                                     (detLimRef,'MODIS DL + FRP VIIRS_as_MODIS, [%g MW]', nmVIIRS_as_MODIS, 'cool'),
                               
#                               (self.BitFields.QA,'QA: 0=cld,1=?,10=clr?,11=clr','Paired'),
#                               (self.BitFields.day_night, '1=day, 0=night','cool'), 
#                               (self.BitFields.sunglint,'0=sunglint','cool'),
#                               (self.BitFields.snow,'0=snow','cool'),
#                               (self.BitFields.land,'0=water,1=coast,10=desert,11=land','Paired'),
#                               (self.lon,'Lon','cool'),
#                               (self.lat,'Lat','cool')
                              ]:
            ax = axes[ixAx]
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
            cs = bmap.scatter(granMODIS.lon[::2,::2].T, granMODIS.lat[::2,::2].T, c=var[::2,::2].T, 
                              s=1, edgecolor=None, norm=norm, cmap=mpl.pyplot.get_cmap(cm))
            cbar = bmap.colorbar(cs,location='bottom',pad="7%")
            cbar.set_label('DetLim / FRP, MW', fontsize=9)
            if len(dicFP[name][0]) > 0:
#                print('Plotting FRP')
                chFires = ', %g fires' % len(dicFP[name][0])
                sort_order = dicFP[name][0].argsort()
                cs2 = bmap.scatter(dicFP[name][1][sort_order], dicFP[name][2][sort_order], 
                                   c=dicFP[name][0][sort_order], s=30, 
                                   edgecolor='black', norm=norm, cmap=cm)  #mpl.pyplot.get_cmap('rainbow'))
#                cbar2 = bmap.colorbar(cs2,location='right',pad="7%")
#                cbar2.set_label('FRP, MW', fontsize=9)
                ax.set_title(chTxt % np.sum(dicFP[name][0][sort_order]), fontsize=10)
            else:
                ax.set_title(chTxt % 0, fontsize=10)
            ixAx += 1

        # If FRP added, remove labels on the right-hand-side map
#        if self.nFires > 0: #FP_frp.shape[0] > 0:
        axes[1].yaxis.set_ticklabels([])
        mpl.pyplot.suptitle(os.path.split(chOutFNm)[1].strip('.png') + chFires, fontsize=14)
        mpl.pyplot.savefig(chOutFNm,dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()
    
    
    ####################################################################################
    
def process_log_files(chV2M_logs, chFNmOut, log_summary, ifMODIS_as_VIIRS):
    #
    # Reads the log file, draws a scatterplot and makes a its summary
    # File content looks like this:
    #
    # 20200121-2335: FRP for MYD14.A2020021.2335.061.2020326073409.hdf: MODIS=0, MODIS_as_VIIRS=0, VIIRS=9.14102, VIIRS_as_MODIS=0
    #
    FRP_MODIS = []
    if ifMODIS_as_VIIRS: FRP_MODIS_as_VIIRS = []
    FRP_VIIRS = []
    FRP_VIIRS_as_MODIS = []
    datesMODIS = []
    log_summary.log('date    FRP MODIS   FRP MODIS_as_VIIRS   FRP VIIRS    FRP VIIRS_AS_MODIS')
    for chV2M_log in chV2M_logs:
        with open(chV2M_log,'r') as fIn:
            for line in fIn:
                if ':' in line:
                    if line.split(':')[1].strip().startswith('FRP for'):
                        log_summary.log(line)
                        flds = line.split()
                        if ifMODIS_as_VIIRS:
                            FRP_MODIS.append(np.float32(flds[-4].split('=')[-1].strip(',')))
                            FRP_MODIS_as_VIIRS.append(np.float32(flds[-3].split('=')[-1].strip(',')))
                            FRP_VIIRS.append(np.float32(flds[-2].split('=')[-1].strip(',')))
                            FRP_VIIRS_as_MODIS.append(np.float32(flds[-1].split('=')[-1]))
                            datesMODIS.append(dt.datetime.strptime(
                                ':'.join([flds[-5].split('.')[-5], flds[-5].split('.')[-4]]),'A%Y%j:%H%M'))
                        else:
                            FRP_MODIS.append(np.float32(flds[-3].split('=')[-1].strip(',')))
                            FRP_VIIRS.append(np.float32(flds[-2].split('=')[-1].strip(',')))
                            FRP_VIIRS_as_MODIS.append(np.float32(flds[-1].split('=')[-1]))
                            datesMODIS.append(dt.datetime.strptime(
                                ':'.join([flds[-4].split('.')[-5], flds[-4].split('.')[-4]]),'A%Y%j:%H%M'))

#                        log_summary.log('%s   %g   %g   %g   %g' % 
#                                        (datesMODIS[-1], FRP_MODIS[-1], FRP_MODIS_as_VIIRS[-1],
#                                         FRP_VIIRS[-1], FRP_VIIRS_as_MODIS[-1]))
    #
    # Draw the collected granule totals. Make drawings against MODIS FRP and MODIS_as_VIIRS FRP
    #
#    for srcMOD, chTitle in [(FRP_MODIS,'MODIS native FRP, [MW]: FRP MODIS_native=%iTW, MODIS_by_VIIRS_OOp=%iTW' % 
#                             (np.round(np.sum(FRP_MODIS)*1e-6),np.round(np.sum(FRP_MODIS_as_VIIRS)*1e-6))),
#                            (FRP_MODIS_as_VIIRS, 'MODIS native FRP, [MW]: FRP MODIS_native=%iTW, MODIS_by_VIIRS_OOp=%iTW' % 
#                             (np.round(np.sum(FRP_MODIS)*1e-6),np.round(np.sum(FRP_MODIS_as_VIIRS)*1e-6)))]:
    fig, ax = mpl.pyplot.subplots(1, figsize=(8,8))
    if ifMODIS_as_VIIRS:
         p1 = ax.scatter(FRP_MODIS_as_VIIRS, FRP_VIIRS, s=1, label='VIIRS native fires\n tFRP=%iTW, r=%3.2g' % 
                         (np.round(np.sum(FRP_VIIRS)*1e-6), np.corrcoef(FRP_MODIS, FRP_VIIRS)[0,1])) #, marker='.')
         p2 = ax.scatter(FRP_MODIS_as_VIIRS, FRP_VIIRS_as_MODIS, s=1, label='VIIRS after MODIS-OOp\n tFRP=%iTW, r=%3.2g' % 
                         (np.round(np.sum(FRP_VIIRS_as_MODIS)*1e-6), np.corrcoef(FRP_MODIS, FRP_VIIRS_as_MODIS)[0,1])) #, marker='.')
    else:
         p1 = ax.scatter(FRP_MODIS, FRP_VIIRS, s=2, label='VIIRS native \n tFRP=%iTW, r=%3.2g' % 
                         (np.round(np.sum(FRP_VIIRS)*1e-6), np.corrcoef(FRP_MODIS, FRP_VIIRS)[0,1])) #, marker='.')
         p2 = ax.scatter(FRP_MODIS, FRP_VIIRS_as_MODIS, s=2, label='VIIRS after MODIS-OOp\n tFRP=%iTW, r=%3.2g' % 
                         (np.round(np.sum(FRP_VIIRS_as_MODIS)*1e-6), np.corrcoef(FRP_MODIS, FRP_VIIRS_as_MODIS)[0,1])) #, marker='.')
    p3=ax.plot([0,1e10],[0.,1e10],linestyle='--',marker='',c='gray')
    if ifMODIS_as_VIIRS:
        ax.set_xlabel('MODIS_as_VIIRS FRP, [MW]\n FRP MODIS_native=%iTW, MODIS_by_VIIRS_OOp=%iTW' % 
                      (np.round(np.sum(FRP_MODIS)*1e-6),np.round(np.sum(FRP_MODIS_as_VIIRS)*1e-6)))
    else:
        ax.set_xlabel('MODIS FRP, [MW]: FRP MODIS_native=%iTW' % (np.round(np.sum(FRP_MODIS)*1e-6)))
    ax.set_ylabel('VIIRS / VIIRS_as_MODIS FRP, [MW]')
    ax.set_xlim(-1,np.max(FRP_VIIRS)*1.1)
    ax.set_ylim(-1,np.max(FRP_VIIRS)*1.1)
    ax.set_xscale('asinh',linear_width=10, base=10)
    ax.set_yscale('asinh',linear_width=10, base=10)
#    ax.set_xscale('symlog', linthresh=5, linscale=0.1)  # create artifacts at log-lin connection
#    ax.set_yscale('symlog', linthresh=5, linscale=0.1)
    ax.legend(labelcolor='linecolor', loc=4)
    ax.grid()
    ax.set_title('VIIRS native and MODIS-OOp-processed fires vs MODIS fires\none dot is total FRP of one MODIS granule, Ngr=%i' % len(FRP_MODIS))
    mpl.pyplot.savefig(chFNmOut, dpi=400)
    mpl.pyplot.clf()
    mpl.pyplot.close()
    print('MODIS granules with fires: ', len(FRP_MODIS), chFNmOut)
    

#######################################################################################################        
        
if __name__ == '__main__':
    print('Hi')

    ifCountGranules = False
    ifVIIRS_2_MODIS_grid = False
    ifVIIRS_2_MODIS_granules = False
    ifVIIRS_2_MODIS_granules_summary = True
    check_timing = False
    ifTestFld = False
    
    
    chDirMODIS_raw = 'd:\\data\\satellites\\MODIS\\'
    chDirWrk = 'd:\\results\\fires\\IS4FIRES_v3_0_MODIS'
    chDirMetadata = 'd:\\project\\fires\\metadata'

    chMetadataFNm = os.path.join(chDirMetadata,
#                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat.ini'),
#                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_clean_BC_OC_filled.txt'),
#                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_CB5_modified_halo_BC_OC.ini')
                                'fire_metadata_ecodata_peat_VIIRS_experim_10MW_only_PM_FRP_continents_v22.ini')    

    gridsAr = [#(gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_3_0.txt')),'glob_3_0'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_2_0.txt')),'glob_2_0'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_1_0.txt')),'glob_1_0'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_0_5.txt')),'glob_0_5'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_0_1.txt')),'glob_0_1'),
             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_0_03.txt')),'glob_0_03'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_0_02.txt')),'glob_0_02'),
#             (gridtools.fromCDOgrid(os.path.join(chDirMetadata,'grd_glob_0_01.txt')),'glob_0_01')
##            (os.path.join(chDirMetadata,'GRIP4_grid_global.txt'),'GRIP4_grd_glob'),
               ]
    if len(sys.argv) > 1:
        print(sys.argv)
        grids = [gridsAr[int(sys.argv[1])]]
    else:
        grids = gridsAr

    #========================================================================================

    if ifCountGranules:
        #
        # Goes through the list of log files and find the granules that have not been processed
        # due to a process crush
        #
        dirOut = 'd:\\results\\fires\\OOP_VIIRS_2_MODIS'
        chV2M_logs = glob.glob(os.path.join(dirOut, 'log_VIIRS_2_MODIS_*_full.txt'))
        dayStart = dt.datetime(2020,1,1)
        dayEnd = dt.datetime(2020,7,14)
        nGrans_tot = np.round((dayEnd - dayStart) / (spp.one_minute * 5)).astype(int)
        timeGrans = np.array(list((dayStart + i * spp.one_minute * 5 for i in range(nGrans_tot))))
        gransDone_ = []
        ifExists = np.zeros(shape=(len(timeGrans)), dtype=np.int8)
        for fNm in chV2M_logs:
            print(fNm)
            with open(fNm,'r') as fIn:
                for line in fIn:
                    if '\\MYD14.A2020' in line:
                        if 'Weird' in line: continue
                        fn_parts = os.path.split(line.split()[2])[-1].split('.')
                        # MYD14.A2020197.0600.061.2020337161400.hdf
                        gransDone_.append(dt.datetime.strptime(fn_parts[1],'A%Y%j') +
                                         (dt.datetime.strptime(fn_parts[2],'%H%M') - dt.datetime(1900,1,1)))
        gransDone = sorted(gransDone_)
        idxGranTimes = np.searchsorted(timeGrans, gransDone).astype(int)
        idxGranTimes[idxGranTimes == len(timeGrans)] = len(timeGrans)-1  # out-of-array search is to be cut
        ifExists = (timeGrans[idxGranTimes] == gransDone)
        print('Total times %i, found missing granules: %i' % (len(timeGrans), np.sum(ifExists)))
    else: 
        gransDone = None
                

    #====================================================================================

    if ifVIIRS_2_MODIS_grid or ifVIIRS_2_MODIS_granules:
        #
        # Take VIIRS data and apply MODIS OOp, then compare with MODIS
        #
        years = [2020]
        chFireRecordsDir = 'd:\\results\\fires\\fire_records_<SAT>'
        chMODIS_rawFRP_templ = 'd:\\data\\satellites\\MODIS\\MYD14\\%Y\\%Y%m%d\\MYD14.A%Y%j.%H%M.061.*.hdf'
        chMODIS_rawGEO_templ = 'd:\\data\\satellites\\MODIS\\MYD35\\%Y\\%Y%m%d\\MYD35_L2.A%Y%j.%H%M.061.*.hdf'
        chVIIRS_rawFRP_templ = 'd:\\data\\satellites\\VIIRS\\VNP14IMG-5200\\%Y%m%d\\VNP14IMG.A%Y%j.%H%M.002.*.nc'
        chVIIRS_rawGEO_templ = 'd:\\data\\satellites\\VIIRS\\VNP03IMG-5200\\%Y%m%d\\VNP03IMG.A%Y%j.%H%M.002.*.nc4'
        dirOut = 'd:\\results\\fires\\OOP_VIIRS_2_MODIS'
        spp.ensure_directory_MPI(dirOut)
        ifDrawGranules = False
        chNow = dt.datetime.now().strftime('%Y%m%d_%H%M')
        log = spp.log(os.path.join(dirOut,'log_VIIRS_2_MODIS_%s_MPI%03i_full.txt' % (chNow, mpirank_loc)))
        OOp = FRP_OOp_pixel.FRP_observation_operator_pixel('MODIS', log)

        for grid_def in grids:
            log.log('VIIRS 2 MODIS, output grid: ' + grid_def[1])
            for year in [2020]:
                log.log('year: %g, getting fire reccords...' % year)
                #
                # get VIIRS and MODIS data
                #
                grid_UTC_recs = 'glob_3_0'
                chFireRecordFNm_yr_VIIRS = os.path.join(chFireRecordsDir.replace('<SAT>','VIIRS'),
                                                        'fire_records_%s_%s_UTC_QA_%g0101_%g1231.nc4' % 
                                                        (grid_UTC_recs,'VIIRS', year, year))
                recsTmp = fire_records.fire_records(log).from_nc(chFireRecordFNm_yr_VIIRS)
                ifSNP = recsTmp.satellite == b'N'
                fRecs_VIIRS = recsTmp.subset_fires(ifSNP)
                print('VIIRS total fire records, FRP:', recsTmp.nFires, np.sum(recsTmp.FRP),
                      'SNP: ', fRecs_VIIRS, np.sum(fRecs_VIIRS.FRP))
                 
                chFireRecordFNm_yr_MODIS = os.path.join(chFireRecordsDir.replace('<SAT>','MODIS'),
                                                        'fire_records_%s_%s_UTC_QA_%g0101_%g1231.nc4' % 
                                                        (grid_UTC_recs,'MODIS', year, year))
                fRecs_MODIS = fire_records.fire_records(log).from_nc(chFireRecordFNm_yr_MODIS)
                #
                # Converter itself
                #
                if ifVIIRS_2_MODIS_grid:
                    #
                    # Gridded OOp application
                    #
                    VIIRS_2_MODIS_grid(OOp, grid_def[0], grid_def[1], fRecs_VIIRS, fRecs_MODIS,
                                       year, dirOut, log)
                    
                elif ifVIIRS_2_MODIS_granules:
                    #
                    # Granule-level pixel OOp application
                    #
                    today = dt.datetime(year,1,1)
                    iProcess = 0
                    while today < dt.datetime(year+1,1,1):
                        # Parallelization
                        iProcess += 1
                        if np.mod(iProcess-1, mpisize_loc) != mpirank_loc: 
                            today += spp.one_day
                            continue

                        VIIRS_2_MODIS_granules(fRecs_VIIRS.subset_time(today, today + spp.one_day),
                                               chMODIS_rawFRP_templ, chMODIS_rawGEO_templ, 
                                               chVIIRS_rawFRP_templ, chVIIRS_rawGEO_templ, 
                                               today, ifDrawGranules, dirOut, log, 
                                               gransDone)
                        today += spp.one_day


    #====================================================================================

    if ifVIIRS_2_MODIS_granules_summary:
        #
        # Reads the log file and draws a scatterplot, prints statistics, etc.
        #
        dirOut = 'd:\\results\\fires\\OOP_VIIRS_2_MODIS'
        
        # Asymmetric
#        chV2M_logs = glob.glob(os.path.join(dirOut, 'asymmetric', 'log_VIIRS_2_MODIS_*_full_pics.txt'))
#        process_log_files(chV2M_logs, os.path.join(dirOut, 'asymmetric_att1.png'),
#                          spp.log(os.path.join(dirOut, 'asymmetric', 'log_VIIRS_2_MODIS_summary.txt')),
#                          False)
        # Symmetric
        chV2M_logs = glob.glob(os.path.join(dirOut, 'log_VIIRS_2_MODIS_*_full.txt'))
#        chV2M_logs = [os.path.join(dirOut,'symmetric_att1_dump_copy.txt')]
        process_log_files(chV2M_logs, os.path.join(dirOut, 'symmetric_att1.png'),
                          spp.log(os.path.join(dirOut, 'log_VIIRS_2_MODIS_summary.txt')), 
                          True)

    #==========================================================================================
    
    if check_timing:
        #
        # Which of the VIIRS satellites are closer to AQUA overpasses?
        #
        dirIn = 'c:\\results\\fires'
        dirOut = 'd:\\results\\fires\\OOP_VIIRS_2_MODIS'
        log = spp.log(os.path.join(dirOut,'check_timing.log'))
        year = 2020
        ranges = np.arange(0.0, 24.0, 0.1)
        
        frecsMODIS = fire_records.fire_records(log).from_nc(os.path.join(dirIn,'fire_records_MODIS',
                                                                         'fire_records_glob_3_0_MODIS_LST_QA_%i0101_%i1231.nc4' % (year, year)))
        histMODIS = np.histogram(np.mod(frecsMODIS.time/3600, 24.0), ranges)
        print('MODIS fires: ', frecsMODIS.nFires) #, frecsMODIS.satellite)
        # AQUA
        maskAQUA = frecsMODIS.satellite == b'Y'
        frecsAQUA = frecsMODIS.subset_fires(maskAQUA)
        histAQUA = np.histogram(np.mod(frecsAQUA.time/3600, 24.0), ranges)
        print('AQUA fires: ', frecsAQUA.nFires)
        
        frecsVIIRS = fire_records.fire_records(log).from_nc(os.path.join(dirIn,'fire_records_VIIRS',
                                                                         'fire_records_glob_3_0_VIIRS_LST_QA_%i0101_%i1231.nc4' % (year, year)))
        histVIIRS = np.histogram(np.mod(frecsVIIRS.time/3600, 24.0), ranges)
        print('VIIRS fires: ', frecsVIIRS.nFires) #, frecsVIIRS.satellite)
        # SNP
        maskSNP = frecsVIIRS.satellite == b'N'
        frecsSNP = frecsVIIRS.subset_fires(maskSNP)
        histSNP = np.histogram(np.mod(frecsSNP.time/3600, 24.0), ranges)
        print('SNP fires: ', frecsSNP.nFires)
        # NOAA
        maskNOAA = frecsVIIRS.satellite == b'1'
        frecsNOAA = frecsVIIRS.subset_fires(maskNOAA)
        histNOAA= np.histogram(np.mod(frecsNOAA.time/3600, 24.0), ranges)
        print('NOAA fires: ', frecsNOAA.nFires)
        
        fig, ax = mpl.pyplot.subplots(1, figsize=(8,6))
        p1 = ax.plot(ranges[:-1], histAQUA[0] / frecsAQUA.nFires, label='AQUA', marker='.')
        p2 = ax.plot(ranges[:-1], histSNP[0] / frecsSNP.nFires, label='SNP, r=%3.2g' % np.corrcoef(histAQUA[0],histSNP[0])[0,1], marker='.')
        p3 = ax.plot(ranges[:-1], histNOAA[0] / frecsNOAA.nFires, label='NOAA, r=%3.2g' % np.corrcoef(histAQUA[0],histNOAA[0])[0,1], marker='.')
        ax.set_title('Diurnal variation of total global FRP, local solar time')
        ax.set_xlabel('hours')
        ax.set_ylabel('fraction of fires')
        ax.set_xlim(0,24)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
#        axes.set_ylim(0,1000) #np.max(FRP_VIIRS))
        ax.legend(loc=2)
        ax.grid()
        mpl.pyplot.savefig(os.path.join(dirOut,'histograms_MODIS_VIIRS_timing.png'), dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()
        #
        # And histograms of FRP
        #
        print('MODIS min/max:', np.min(frecsMODIS.FRP), np.max(frecsMODIS.FRP))
        print('AQUA min/max:', np.min(frecsAQUA.FRP), np.max(frecsAQUA.FRP))
        print('VIIRS min/max:', np.min(frecsVIIRS.FRP), np.max(frecsVIIRS.FRP))
        print('SNP min/max:', np.min(frecsSNP.FRP), np.max(frecsSNP.FRP))
        print('NOAA min/max:', np.min(frecsNOAA.FRP), np.max(frecsNOAA.FRP))
        histMODIS_FRP = np.histogram(frecsMODIS.FRP,
                                     bins=np.logspace(np.log10(np.min(frecsMODIS.FRP)),
                                                      np.log10(np.max(frecsMODIS.FRP)), 100))
        histVIIRS_FRP = np.histogram(frecsVIIRS.FRP,
                                     bins=np.logspace(np.log10(np.min(frecsVIIRS.FRP)),
                                                      np.log10(np.max(frecsVIIRS.FRP)), 100))
        histAQUA_FRP = np.histogram(frecsAQUA.FRP,
                                    bins=np.logspace(np.log10(np.min(frecsAQUA.FRP)),
                                                     np.log10(np.max(frecsAQUA.FRP)), 100))
        histSNP_FRP = np.histogram(frecsSNP.FRP,
                                   bins=np.logspace(np.log10(np.min(frecsSNP.FRP)),
                                                    np.log10(np.max(frecsSNP.FRP)), 100))
        histNOAA_FRP = np.histogram(frecsNOAA.FRP,
                                    bins=np.logspace(np.log10(np.min(frecsNOAA.FRP)),
                                                     np.log10(np.max(frecsNOAA.FRP)), 100))
        
        fig, axes = mpl.pyplot.subplots(1,2, figsize=(14,7))
        p1 = axes[0].semilogx(histMODIS_FRP[1][:-1], histMODIS_FRP[0], label='MODIS', marker='.')
        p2 = axes[0].semilogx(histVIIRS_FRP[1][:-1], histVIIRS_FRP[0], label='VIIRS', marker='.')
        p3 = axes[0].semilogx(histSNP_FRP[1][:-1], histSNP_FRP[0], label='SNP', marker='.')
        p4 = axes[0].semilogx(histNOAA_FRP[1][:-1], histNOAA_FRP[0], label='NOAA', marker='.')
        axes[0].set_title('FRP distribution density, %i' % year)
        axes[0].set_xlabel('FRP, [MW]')
        axes[0].set_ylabel('Nbr of fires')
#        ax.set_xlim(0,24)
#        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
#        axes.set_ylim(0,1000) #np.max(FRP_VIIRS))
        axes[0].legend(loc=1)
        axes[0].grid()
        
        p5 = axes[1].loglog(histMODIS_FRP[1][:-1], histMODIS_FRP[0], label='MODIS', marker='.')
        p6 = axes[1].loglog(histVIIRS_FRP[1][:-1], histVIIRS_FRP[0], label='VIIRS', marker='.')
        p7 = axes[1].loglog(histSNP_FRP[1][:-1], histSNP_FRP[0], label='SNP', marker='.')
        p8 = axes[1].loglog(histNOAA_FRP[1][:-1], histNOAA_FRP[0], label='NOAA', marker='.')
        axes[1].set_title('FRP distribution density, %i' % year)
        axes[1].set_xlabel('FRP, [MW]')
        axes[1].set_ylabel('Nbr of fires')
#        ax.set_xlim(0,24)
#        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
#        axes.set_ylim(0,1000) #np.max(FRP_VIIRS))
        axes[1].legend(loc=1)
        axes[1].grid()
        mpl.pyplot.savefig(os.path.join(dirOut,'histograms_MODIS_VIIRS_FRP_%i.png' % year), dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()
        

    if ifTestFld:
        #
        # create the directory for temporary log files
        # clean the old mess, if any
        if os.path.exists(os.path.join(chDirWrk, 'chDirTmp_ready')): 
            os.removedirs(os.path.join(chDirWrk, 'chDirTmp_ready'))
        if os.path.exists(os.path.join(chDirWrk, 'chDirTmp.inf')): 
            os.remove(os.path.join(chDirWrk, 'chDirTmp.inf'))
        # make directory
        chDirTmp = dt.datetime.now().strftime('tmp_%Y%m%d_%H%M')
        os.makedirs(chDirTmp)
        print('Temporary directory: ', chDirTmp)
        
        for grid_def in grids:
            #
            # only rank 0 is allowed to make directories 
            for d in [os.path.join(chDirWrk, grid_def[1]), 
                      os.path.join(chDirWrk, grid_def[1] +'_LST'),
                      os.path.join(chDirWrk, grid_def[1] + '_LST_daily')]:
                spp.ensure_directory_MPI(d)
            grid = gridtools.fromCDOgrid(grid_def[0])
            #
            # Initialise the main fire emission model
            # It is assumed that the first MODIS processing is already done and nc4 files exist
            #
            OOp = FRP_OOp_grid.FRP_observation_operator_grid(
                                                os.path.join(chDirMODIS_raw,
                                                             'MxD14_coll_6_extract', '%Y', '%Y.%m.%d', 
                                                             'MxD14.A%Y%j.%H%M.006.*.hdf.nc4'), 
                                                os.path.join(chDirMODIS_raw,
                                                             'MxD35_L2_extract', '%Y', '%j',
                                                             'MxD35_L2.A%Y%j.%H%M.061.*hdf_extract.nc4'),
                                                'MODIS', (grid,grid_def[1]),
                                                spp.log(os.path.join(chDirWrk,chDirTmp,'try_OOp.log')))
            # Test field
            #
            mdlFRP = np.ones(shape=(24, grid.ny, grid.nx),dtype=np.float32) * 200.0
            
            mdlFRP_obs, DL, DL_clrsky, nodat = OOp.observe_map(mdlFRP, dt.datetime(2003,2,10, 12), 0.0, None)
    
            fOut = silamfile.open_ncF_out(os.path.join(chDirWrk,'OOp_output.nc4'),
                                          'NETCDF4', grid, silamfile.SilamSurfaceVertical(),
                                          dt.datetime(2003,2,10), 
                                          np.array(list((dt.datetime(2003,2,10) + i * spp.one_hour 
                                                         for i in range(24)))),
                                          [], ['FRP_obs', 'FRP_DL', 'FRP_DL_clrsky', 'noData'], 
                                          {'FRP_obs':'MW', 'FRP_DL':'MW', 'FRP_DL_clrsky':'MW', 'noData':''},
                                          -999999, True, 3)
            fOut.variables['FRP_obs'][:,:,:] = mdlFRP_obs[:,:,:]
            fOut.variables['FRP_DL'][:,:,:] = DL[:,:,:]
            fOut.variables['FRP_DL_clrsky'][:,:,:] = DL_clrsky[:,:,:]
            fOut.variables['noData'][:,:,:] = nodat[:,:,:]
            fOut.close()

        
        
        
    