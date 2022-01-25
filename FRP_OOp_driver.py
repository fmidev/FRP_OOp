'''

Driver for the FRP observation operator
The top-most module for the FRP OOp set

Created on 14.1.2022

@author: sofievm
'''

import os, sys, numpy as np, datetime as dt
from support import suntime
from toolbox import supplementary as spp, gridtools, silamfile
from src import FRP_OOp_grid

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
try:
    # slurm loader environment
    mpirank = int(os.getenv("SLURM_PROCID",None))
    mpisize = int(os.getenv("SLURM_NTASKS",None))
    chMPI = '_mpi%03g' % mpirank
    comm = None
    print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize = comm.size
        mpirank = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank
        print ('MPI operation, mpisize=', mpisize, chMPI)
    except:
        print ("mpi4py failed, single-process operation")
        mpisize = 1
        mpirank = 0
        chMPI = ''
        comm = None


#
# The top function: observe
#
def observe_FRP_gridded(mdlFRP, grid, timeUTC, chInstrument, chSatFNmTempl, ifNight_=None):
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
        
if __name__ == '__main__':
    print('Hi')
    
    chDirMODIS_raw = 'f:\\data\\MODIS_raw_data\\'
    chDirWrk = 'f:\\project\\fires\\IS4FIRES_v3_0_grid_FP'
    chDirMetadata = 'd:\\project\\fires\\forecasting_v2_0'
    chDirEcodata = 'd:\\data\\emis\\fires'
    # puhti
#    chDirMODIS_raw = '/fmi/scratch/project_2001411/data/MODIS_raw_data'
#    chDirWrk = '/fmi/scratch/project_2001411/fires/IS4FIRES_v3_0_grid_FP'
#    chDirMetadata = '/fmi/projappl/project_2001411/fires'
#    chDirEcodata = chDirMetadata

    chMetadataFNm = os.path.join(chDirEcodata,
#                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat.ini'),
#                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_clean_BC_OC_filled.txt'),
                              'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_CB5_modified_halo_BC_OC.ini')    

    gridsAr = [#(os.path.join(chDirMetadata,'grd_EU_0_05.txt'),'EU_0_05'),      # 1
               #(os.path.join(chDirMetadata,'grd_EU_0_1.txt'),'EU_0_1'),           # 2
               #(os.path.join(chDirMetadata,'grd_EU_0_2.txt'),'EU_0_2'),           # 2
               #(os.path.join(chDirMetadata,'grd_EU_0_5.txt'),'EU_0_5'),           # 2
               #(os.path.join(chDirMetadata,'grd_EU_1_0.txt'),'EU_1_0'),           # 2
               #(os.path.join(chDirMetadata,'grd_EU_2_0.txt'),'EU_2_0'),           # 2
#               (os.path.join(chDirMetadata,'grd_glob_0_1_180.txt'),'glob_0_1'),
               (os.path.join(chDirMetadata,'grd_glob_3_0.txt'),'glob_3_0'),       # 2
               #(os.path.join(chDirMetadata,'grd_glob_1_0.txt'),'glob_1_0'),       # 2
               #(os.path.join(chDirMetadata,'grd_glob_2_0.txt'),'glob_2_0'),       # 2
               #(os.path.join(chDirMetadata,'grd_glob_0_5.txt'),'glob_0_5'),           # 0
               #(os.path.join(chDirMetadata,'GRIP4_grid_global.txt'),'GRIP4_grd_glob'),
               ]
    if len(sys.argv) > 1:
        print(sys.argv)
        grids = [gridsAr[int(sys.argv[1])]]
    else:
        grids = gridsAr

    #
    # create the directory for temporary log files
    #
    if mpirank == 0: 
        # clean the old mess, if any
        if os.path.exists(os.path.join(chDirWrk, 'chDirTmp_ready')): 
            os.removedirs(os.path.join(chDirWrk, 'chDirTmp_ready'))
        if os.path.exists(os.path.join(chDirWrk, 'chDirTmp.inf')): 
            os.remove(os.path.join(chDirWrk, 'chDirTmp.inf'))
        # make directory
    chDirTmp = dt.datetime.now().strftime('tmp_%Y%m%d_%H%M')
    spp.ensure_directory_MPI(os.path.join(chDirWrk,chDirTmp))
    print('Temporary directory: ', chDirTmp)
    
#    logMain = spp.log(os.path.join(chDirWrk, chDirTmp, '_run_log_main_%03i.txt' % mpirank))

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
        #
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

