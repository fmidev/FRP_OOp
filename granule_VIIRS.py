'''
Created on 23.5.2022

 The class holding the information on a single swatch of the VIIRS satellite instrument 

Functions available:
- __init__:   Sets abasic file names 
- get_pixel_size:  basic VIIRS geometry
- MODIS_1km_geo_from_5km: Takes the so-called reduced-resolution fields of longitude and latitude
                          and makes full-resolutionn ones for 1km
- unpack_MxD03_byte_1: Bit fields within each byte are numbered from the left: 7, 6, 5, 4, 3, 2, 1, 0.
- unpack_MxD14_QA:     unpacks the quality field. NotImplemented
- pick_granule_data_IS4FIRES_v3_0: gets the needed data from MxD14 and MxD35
######- detection_limit;     returns the detection limit map as a function of the day/night flag and pixel size
- pick_granule_data_IS4FIRES_v2_0: gets data needed for old IS4FIRES version from MOD14
- write_granule_IS4FIRES_v2_0:  write the granule down as IS4FREIS 2.0 need
- draw_granule:        Draws area covered by this swath, with a bit free space around:

@authors: M.Sofiev, R.Kouznetsov 
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
import granule__basic as gb, fire_records
from toolbox import drawer, supplementary as spp, silamfile, gridtools, util

#
# MPI may be tricky: in puhti it loads fine but does not work
# Therefore, first check for slurm loader environment. Use it if available
#
try:
    # slurm loader environment
    mpirank_loc = int(os.getenv("SLURM_PROCID",None))
    mpisize_loc = int(os.getenv("SLURM_NTASKS",None))
    chMPI = '_mpi%03g' % mpirank_loc
    comm = None
    print('SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize_loc)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize_loc = comm.size
        mpirank_loc = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank_loc
        print ('MPI operation, mpisize=', mpisize_loc, chMPI)
    except:
        print ("mpi4py failed, single-process operation")
        mpisize_loc = 1
        mpirank_loc = 0
        chMPI = ''
        comm = None

# FORTRAN module
try:
    import f_granule_viirs
    ifFortranOK = True
except:
    # Attention: sizes of the arrays must be at the end
    fortran_code_VIIRS = '''

!========================================================================================
  
  subroutine f_upsample_fld (fIn, fOut, scanSz_In_tr, idx_segments_HiR, &
                           & pixel_nbr_scaling_along_track, pixel_nbr_scaling_along_scan, &
                           & nPixT_out, nPixS_out, nPixT_in, nPixS_in)
  !
  ! Reverse to downsample: inter-/extra-polates the low-resolution field to high-resolving one
  ! Procedure is scan-wise and aggregation-wise, includes two-steps: 
  ! (i) each low-res line along-track-within-scan is filled with linear inter-extrapolation, 
  ! (ii) these lines are inter-extrapolated along the scan aggregation segments, not allowing
  !      interpolation across the border ofg these segments
  !
  implicit none

  ! Imported parameters
  real*4, dimension(0:nPixT_in-1, 0:nPixS_in-1), intent(in) :: fIn
  real*4, dimension(0:nPixT_out-1, 0:nPixS_out-1), intent(out) :: fOut
  integer*4, dimension(0:5), intent(in) :: idx_segments_HiR
  integer*4, intent(in) :: pixel_nbr_scaling_along_track, pixel_nbr_scaling_along_scan 
  ! sizes at the end
  integer*4, intent(in) :: scanSz_In_tr, nPixT_out, nPixS_out, nPixT_in, nPixS_in

  ! Local variables
  integer :: iSc, iTmp, iLR_0, iLR_1, iSeg, scanSz_Out_tr
  real*4, dimension(0:scanSz_In_tr-1) :: pos_tr    ! lowres scan size along trac
  real*4, dimension(0:nPixS_in-1) :: pos_sc        ! full width of lowres scan
  integer*4, dimension(0:5) :: idx_segments_LoR    ! low-resolution indices of segments
  ! intermediate from step 1: highres along track, lowres along scan
  real*4, dimension(0:scanSz_In_tr*pixel_nbr_scaling_along_track-1, 0:nPixS_in-1) :: line
  real*4, dimension(0:nPixS_in-1) :: slope                  ! slope for making the line
  !
  ! nullify output map
  fOut(:,:) = 0.0
  

!  print *, 'fIn', minval(fIn), maxval(fIn), sum(fIn)
!  print *, 'scanSz_In_tr', scanSz_In_tr
!  print *, 'pixel_nbr_scaling_along_track, pixel_nbr_scaling_along_scan, nPixT_out, nPixS_out, nPixT_in, nPixS_in', &
!          & pixel_nbr_scaling_along_track, pixel_nbr_scaling_along_scan, nPixT_out, nPixS_out, nPixT_in, nPixS_in 
  
  
  scanSz_Out_tr = scanSz_In_tr * pixel_nbr_scaling_along_track  ! scan size along track of the output

  ! Positions of low-res-points within the high-res scan
  ! and segment edges in low-res scan
  do iTmp = 0, scanSz_In_tr-1
    pos_tr(iTmp) = pixel_nbr_scaling_along_track / 2.- 0.5 + pixel_nbr_scaling_along_track * iTmp   ! along-track
  end do
  idx_segments_LoR = nPixS_in
  do iTmp = 0, nPixS_in-1
    pos_sc(iTmp) = pixel_nbr_scaling_along_scan / 2.- 0.5 + pixel_nbr_scaling_along_scan * iTmp   ! along-scan
    ! in what segment?
    do iSeg = 0, 4
      if(pos_sc(iTmp) > idx_segments_HiR(iSeg) .and. pos_sc(iTmp) < idx_segments_HiR(iSeg+1))then
        idx_segments_LoR(iSeg) = min(idx_segments_LoR(iSeg), iTmp)
        exit
      endif
    end do
  end do
  
!  print *, 'idx_segments_HiR', idx_segments_HiR
!  print *, 'idx_segments_LoR', idx_segments_LoR
  
  ! Proceed scan-by-scan
  !
  do iSc = 0, nPixT_in / scanSz_In_tr - 1   ! number of scans
!    print *, 'pos_sc', pos_sc
!    print *, 'idx_segments_HiR', idx_segments_HiR
!    print *, 'iScan, range', iSc, iSc*scanSz_Out_tr, (iSc + 1) * scanSz_Out_tr - 1, nPixS_out-1
    !
    ! Fill-in the along-track lines. Note that they are shifted by 0.5 in both directions
    ! Interpolation goes from two reference points. When reaching the further one, shift
    !
    iLR_0 = 0  ! the first low-res reference point
    iLR_1 = 1  ! the second low-res reference point
    
    slope(:) = (fIn(iSc * scanSz_In_tr + iLR_1, :) - fIn(iSc * scanSz_In_tr + iLR_0, :)) &
           & / (pos_tr(iLR_1) - pos_tr(iLR_0))

    do iTmp = 0, scanSz_Out_tr - 1              ! cycle over high-res line along track
      if(iTmp >= pos_tr(iLR_1))then             ! reached the farther ref point?
        iLR_1 = min(iLR_1+1, scanSz_In_tr-1)    ! allow extrapolation at the end
        iLR_0 = iLR_1 - 1
      endif
      line(iTmp,:) = fIn(iSc * scanSz_In_tr + iLR_0, :) + slope(:) * (iTmp - pos_tr(iLR_0))
    end do   ! high-res lines
    !
    ! Step 2: intermeditate lines to the whole scan. The procedure is transpose to the above
    !         and performed segment by segment to avoid interpolation through the borders
    !
    iLR_0 = 0  ! the first low-res reference point
    iLR_1 = 1  ! the second low-res reference point
!    do iTmp = 0, nPixS_out-1                 ! cycle over high-res scan
!      if(iTmp >= pos_sc(iLR_1))then          ! reached the further ref point?
!        iLR_1 = min(iLR_1+1, nPixS_in-1)     ! allow extrapolation at the end
!        iLR_0 = iLR_1 - 1
!      endif

    iSeg = 0
    do iTmp = 0, nPixS_out-1        ! cycle over high-res scan
      if(iTmp >= idx_segments_HiR(iSeg+1))then   ! determine the left border: new segment?
        iSeg = iSeg + 1
        iLR_0 = iLR_0 + 2   ! no interpolation across segments
        iLR_1 = iLR_0 + 1
!        print *, 'New segment, ', iSeg, iLR_0
      endif
      if(iTmp >= pos_sc(iLR_1))then          ! reached the further ref point?
        iLR_1 = min(iLR_1+1, idx_segments_LoR(iSeg+1)-1)  ! extrapolation at the end, not crossing segments
        iLR_0 = iLR_1 - 1
!        print *, 'Reached the further point, may be, switch LR range', iLR_0, iLR_1
      endif
      
!      print *, 'iTmp_2', iTmp, iSeg, idx_segments_HiR(iSeg), idx_segments_HiR(iSeg+1), iLR_0, iLR_1 
      
      
      fOut(iSc * scanSz_Out_tr : (iSc + 1) * scanSz_Out_tr - 1, iTmp) = line(:,iLR_0) &
                                & + (line(:,iLR_1) - line(:,iLR_0)) &
                                & * (iTmp - pos_sc(iLR_0)) / (pos_sc(iLR_1) - pos_sc(iLR_0))
    end do   ! high-res lines
  end do  ! main cycle over scans

end subroutine f_upsample_fld


!============================================================================

subroutine f_get_height(glob_height_m, fxGran, fyGran, granule_true_height, &
                      & nxGlob, nyGlob, nxGran, nyGran)
  !
  ! Picks the height for the granule, which indices in the height field are given
  !
  implicit none
  !
  ! Input parameters
  real*4, dimension(0:nxGlob-1, 0:nyGlob-1), intent(in) :: glob_height_m
  real*4, dimension(0:nxGran-1, 0:nyGran-1), intent(in) :: fxGran, fyGran
  real*4, dimension(0:nxGran-1, 0:nyGran-1), intent(out) :: granule_true_height
  integer, intent(in) :: nxGlob, nyGlob, nxGran, nyGran
  
  ! Local variables
  integer :: ix, iy, iTmp
  
  ! Cycle over the given index arrays picking the corresponding values
  do iy = 0, nyGran-1
    do ix = 0, nxGran-1
      iTmp = nint(fyGran(ix,iy))
      if(iTmp == nyGlob)iTmp = nyGlob-1
      granule_true_height(ix,iy) = glob_height_m(nint(fxGran(ix,iy)), iTmp)  !&
!                                             & max(0, min(nyGlob-1, nint(fyGran(ix,iy)))))
    end do
  end do  ! iy
end subroutine f_get_height


'''

#    from numpy.distutils.fcompiler import new_fcompiler
#    compiler = new_fcompiler(compiler='intel')
#    compiler.dump_properties()

    # Compile the library and, if needed, copy it from the subdir where the compiler puts it
    # to the current directory
    #
    
    vCompiler = np.f2py.compile(fortran_code_VIIRS, modulename='f_granule_viirs', 
                                verbose=1, extension='.f90')
    if vCompiler == 0:
        cwd = os.getcwd()     # current working directory
        print(cwd)
        if os.path.exists(os.path.join('f_granule_viirs','.libs')):
            list_of_files = glob.glob(os.path.join('f_granule_viirs','.libs','*'))
            latest_file = max(list_of_files, key=os.path.getctime)
            shutil.copyfile(latest_file, os.path.join(cwd, os.path.split(latest_file)[1]))
        try: 
            import f_granule_viirs
            ifFortranOK = True
        except:
            print('>>>>>>> FORTRAN failed-2, have to use Python. It will be SLO-O-O-O-O-O-OW')
            ifFortranOK = False
    else:
        print('>>>>>>> FORTRAN failed, have to use Python. It will be SLO-O-O-O-O-O-OW')
        ifFortranOK = False

ifDebug = False

STATUS_SUCCESS = 0
STATUS_BAD_FILE = -1
STATUS_MISSING_FRP_FILE = -2
STATUS_MISSING_GEO_FILE = -2

#=============================================================

def productType(chFNm):
    dicKnownProducts = {'VNP03IMGLL':('VIIRS','auxiliary'), 'VJ103IMG':('VIIRS','auxiliary'),
                        'VNP03IMG':('VIIRS','auxiliary'),   'VNP14IMGLL':('VIIRS','fire'),
                        'VJ114IMG':('VIIRS','fire'),        'VNP14IMG':('VIIRS','fire')}
    for satProd in dicKnownProducts.keys():
        if satProd in chFNm: return dicKnownProducts[satProd]
    return 'unknown'



#################################################################################
#
# The class holding the information on a single swatch of the satellite, 
#
#################################################################################

class granule_VIIRS(gb.granule_basic):

    #=============================================================
    def __init__(self, now_UTC=dt.datetime.utcnow(), 
                 chFRPfilesTempl='', chAuxilFilesTempl='', # chFNm_GlobHeight='',
                 log=None):
        #
        # Initialises the basic object and VIIRS specifics
        #
        self.type = 'VIIRS'
        gb.granule_basic.__init__(self, self.type, now_UTC, chFRPfilesTempl, chAuxilFilesTempl, log,
                                  (log is None) or (not ifDebug))
        self.chFNm_GlobHeight = '' #chFNm_GlobHeight   not needed, really...
        self.glob_height_fIn = None         # not yet. May be, not needed
        self.glob_height_metadata = None
        self.get_pixel_size()        # get the distribution of MODIS swath

    #=============================================================
    def get_FRP_FNm(self):
        try: return self.chFNm_FRP
        except: return None 
    
    #=============================================================
    def get_geo_FNm(self):
        try: return self.chFNm_geo
        except: return None 

    #=============================================================
    def check_file_type(self, ShortName):
        VIIRSShortNames="VNP03IMGLL VJ103IMG VNP03IMG".split()
        if ShortName in VIIRSShortNames:
            return True
        else:
            raise AttributeError("shortName %s not implemented, should be one of [%s]"%
                                 (ShortName, ",".join(VIIRSShortNames))) 
    
    #=============================================================
    def abbrev_1(self, chFNm):
        if os.path.basename(chFNm).startswith('VNP'):
            return 'N'.encode('utf-8')
        elif os.path.basename(chFNm).startswith('VJ1'): 
            return '1'.encode('utf-8')
        else: raise ValueError('Cannot determine the satellite from name:', chFNm)


    #=============================================================

    def get_pixel_size(self):
        #
        # An approximate but quite accurate formula for dS and dT pixel size along the scan 
        # and along the track directions, respectively
        # Presented in Ichoku & Kaufman, IEEE Transactions 2005
        # VIIRS aggregation featres are presented in 
        # Wolfe, R.E., Lin, G., Nishihama, M., Tewari, K.P., Tilton, J.C., Isaacman, A.R., 2013. 
        # Suomi NPP VIIRS prelaunch and on-orbit geometric calibration and characterization. 
        # JGR Atmos. 118. https://doi.org/10.1002/jgrd.50873
        # and
        # Schroeder, W., Oliva, P., Giglio, L., Csiszar, I.A., 2014. The New VIIRS 375 m active 
        # fire detection data product: Algorithm description and initial assessment. Remote
        # sensing of Environment, 143, 85-96, https://doi.org/10.1016/j.rse.2013.12.008 
        # Note certain differences between the numbers reported by different sources. In such cases,
        # Wolfe's numbers are used as more specific and up to the point, Schroeder states values
        # which contradict to their pictures.
        #
        # This sub has been calibrated against the lookup table obtained via personal communication 
        # with Wilfrid Schroeder, 7 Jul 2022, with the following accuracy:
        # 
        # dS error, km: min_LUT-fla, max_LUT-fla, max_relative: -0.000100234 7.53053e-05 0.000261025
        # dT error, km: min_LUT-fla, max_LUT-fla, max_relative: -3.05407e-05 5.09228e-05 9.35962e-05
        # theta error, rad: min_LUT-fla, max_LUT-fla, max_relative: -6.98713e-07 9.58014e-07 0.0020582
        # The relative error is normalised pixel-wise for dS and dT with dS and dT, but for theta
        # it is normalised with the angular pixel size.
        #
        self.h = 825.2      # VIIRS altitude, km. actually 830-365 at equator-poles
        self.N_375 = 6400      # total number of I-band pixels along the scan, note that some are aggregated
        self.N = self.N_375  # for the sake of compatibility
        self.minAngle = np.float64(-56.05555 * np.pi / 180.)   # Roux's fit is accurate for this range
#        self.minAngle = np.float64(-55.891668 * np.pi / 180.)   # Roux's fit is accurate for this range
#        self.minAngle = np.float64(-56.28 * np.pi / 180.)  # Cao ea, doi:10.1002/2013JD020418
#        self.minAngle = np.float64(-56.065 * np.pi / 180.) # in VIIRS ATBD, https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-004_JPSS_ATBD_VIIRS-Geolocation_A.pdf
        self.maxAngle = -self.minAngle
        self.scanSz_375 = 32
        r = gb.R_Earth + self.h
        # Non-aggregated scan geometry
        sz_scan_nadir_NA = 0.3841/3 # Wolfe ea size of nadir pixel along scan
#        sz_scan_nadir_NA = 0.393/3 # Wolfe ea size of nadir pixel along scan
#        sz_scan_nadir_NA = 0.375/3   # Schroeder ea 375., size of nadir pixel along scan
        self.N_NA = 12608            # I-band
        self.theta_NA =  np.array(list( (self.minAngle + i*(self.maxAngle-self.minAngle)/(self.N_NA-1)
                                         for i in range(self.N_NA))))
        aTmp = np.sqrt((gb.R_Earth/ r)**2 - np.square(np.sin(self.theta_NA)))
        self.dS_NA = gb.R_Earth * sz_scan_nadir_NA / self.h * ((np.cos(self.theta_NA)/ aTmp) - 1.0)   # size along scan, no aggregation
        # Aggregate
        # Wolfe ea give the following indices for M-band (starting half of the scan):
        # before aggregation: 1x :  1-640, 2x:  641-1376, 3x: 1377-3152
        # after aggregation:  1x :  1-640, 2x:  641-1008, 3x: 1009-1600
        # for I-band, double all
        #  M:  1    2    3    4    5     6 
        #  I: 1 2  3 4  5 6  7 8  9 10  11 12
        # before aggregation: 1x :  1-1280, 2x:  1281-2752, 3x: 2753-6304
        # after aggregation:  1x :  1-1280, 2x:  1281-2016, 3x: 2017-3200
        self.dS = np.zeros(shape=(self.N_375))
        self.theta = np.zeros(shape=(self.N_375))
        # starting tail: no aggregation
        self.dS[:1280] = self.dS_NA[:1280]   # M-band till 640, I-band till 1281, inclusive
        self.theta[:1280] = self.theta_NA[:1280]
        # starting mid-part
        self.dS[1280:2016] = np.sum(self.dS_NA[1280:2752].
                                    reshape(int((2752-1280)/2),2), axis=1)
        self.theta[1280:2016] = np.mean(self.theta_NA[1280:2752].
                                        reshape(int((2752-1280)/2),2), axis=1)
        # Nadir section
        self.dS[2016:self.N_375-2016] = np.sum(self.dS_NA[2752:self.N_NA-2752].
                                               reshape(int((self.N_NA-2752*2)/3),3), axis=1)
        self.theta[2016:self.N_375-2016] = np.mean(self.theta_NA[2752:self.N_NA-2752].
                                                   reshape(int((self.N_NA-2752*2)/3),3), axis=1)
        # ending mid-part
        self.dS[self.N_375-2016:self.N_375-1280] = np.sum(self.dS_NA[self.N_NA-2752:self.N_NA-1280].
                                                          reshape(int((2752-1280)/2),2), axis=1)
        self.theta[self.N_375-2016:self.N_375-1280] = np.mean(self.theta_NA[self.N_NA-2752:self.N_NA-1280].
                                                              reshape(int((2752-1280)/2),2), axis=1)
        # ending tail
        self.dS[self.N_375-1280:] = self.dS_NA[self.N_NA-1280:]
        self.theta[self.N_375-1280:] = self.theta_NA[self.N_NA-1280:]

        # Store the indices of the aggregation segments
        self.idx_segments = np.array([0, 1280, 2016, self.N_375-2016, self.N_375-1280, self.N_375])
        
        # pixel size along track
#        self.sz_track_nadir = 0.376 # Wolfe ea 376 Schroeder ea ~375. # size of nadir pixel along track
        self.sz_track_nadir = 0.36053 # Wolfe ea 376 Schroeder ea ~375. # size of nadir pixel along track
        self.dT = (r * self.sz_track_nadir / self.h *             # size along track
                   (np.cos(self.theta) - np.sqrt((gb.R_Earth/ r)**2 - np.square(np.sin(self.theta)))))

        # distance from swath centre point
        c = r * np.cos(self.theta) - np.sqrt( r*r* np.square(np.cos(self.theta)) + gb.R_Earth*gb.R_Earth - r*r)
        self.dist_nadir = gb.R_Earth * np.arcsin(c * np.sin(self.theta) / gb.R_Earth)
        self.area = self.dS * self.dT
        # Overlap along the scan direction (bow-tie), without bow-tie deletion
        self.dist_pixels_S = self.dist_nadir[1:] - self.dist_nadir[:-1]   # minus one element
        self.overlap_S = ((self.dS[1:] + self.dS[:-1]) * 0.5 / self.dist_pixels_S - 1)  # fraction
        # overlap along track is based on 3000 km length of a single swatch (approx. value)
        self.overlap_T = self.dT / np.min(self.dT) - 1  # fraction
#        self.area_corr = self.area / (1. - self.overlap_T)  # did not find the use case
        #
        # make a map of overlapping pixels for a single scan
        # cell centers of the edge of the previous scan projected to the current one (-0.5 in the middle of scan)
        edge_prev_scan_along_track = -1 + self.overlap_T * self.scanSz_375/2  # from -1 till 18
        edge_next_scan_along_track = self.scanSz_375 - self.overlap_T * self.scanSz_375/2  # from 13 till 32
        # overlap fraction over the 32-wide scan
        self.overlap_pattern = np.ones(shape=(self.N_375, self.scanSz_375), dtype=np.float32)  # one scan
        for iTrac in range(self.scanSz_375):
            self.overlap_pattern[edge_prev_scan_along_track >= iTrac, iTrac] = 2  # overlap with prev scan
            self.overlap_pattern[edge_next_scan_along_track <= iTrac, iTrac] = 2  # overlap with next scan
            self.overlap_pattern[np.logical_and(edge_next_scan_along_track <= iTrac,
                                                edge_prev_scan_along_track >= iTrac), 
                                                iTrac] = 3  # two overlaps
        

    #======================================================================

    def upsample_fld(self, DS, scanSz, idx_segments, DS_factor_track, DS_factor_scan):
        #
        # Mimics MAS fortran code in numpy
        #
        nTds,nSds = DS.shape
        nT,nS = nTds*DS_factor_track, nSds*DS_factor_scan
    
        dsSegments = idx_segments // DS_factor_scan
        dsScanSz = scanSz // DS_factor_track
        assert( np.all(idx_segments == dsSegments*DS_factor_scan))
        assert( dsScanSz == 2 ) ## Should work also for others, but compatibility with MAS code might be broken
    
        x = np.newaxis ## just alias
        slopeT = np.zeros((nTds,nSds), dtype=np.float32) ##Along-track gradient of variable
        for i in range(dsScanSz):
            if i == 0: ## first DS line in scan
                slopeT[i::dsScanSz,:] = DS[i+1::dsScanSz,:] - DS[i::dsScanSz,:]
            elif i == dsScanSz - 1: ##last DS line in scan
                slopeT[i::dsScanSz,:] = DS[i::dsScanSz,:] - DS[i-1::dsScanSz,:]
            else: ##Central difference
                slopeT[i::dsScanSz,:] = DS[i+1::dsScanSz,:] - DS[i-1::dsScanSz,:] / 2
    
        slopeS = np.zeros((nTds,nSds,2), dtype=np.float32) ## Along-scan gradient of variable
        slopeST = np.zeros((nTds,nSds,2), dtype=np.float32) ## Along-scan gradient of Along-track gradient
        for iSeg in range(len(idx_segments)-1):
            segstart = idx_segments[iSeg] // DS_factor_scan
            segend   = idx_segments[iSeg+1] // DS_factor_scan
            for i in range(segstart,segend): 
                if i == segstart: ## first DS pixel in segment
                    slopeS[:,i,0:2] = DS[:,i+1,x] - DS[:,i,x]
                    slopeST[:,i,0:2] = slopeT[:,i+1,x] - slopeT[:,i,x]
                elif i == segend - 1: ##last DS pixel in segment
                    slopeS[:,i,0:2] = DS[:, i, x] - DS[:,i-1,x] 
                    slopeST[:,i,0:2] = slopeT[:, i, x] - slopeT[:,i-1,x] 
                else: ##Central difference
                    slopeS[:,i,0] = DS[:, i]   - DS[:,i-1]
                    slopeS[:,i,1] = DS[:, i+1] - DS[:,i]
                    slopeST[:,i,0] = slopeT[:, i]   - slopeT[:,i-1]
                    slopeST[:,i,1] = slopeT[:, i+1] - slopeT[:,i]
    
        stencilT = np.arange(DS_factor_track, dtype=np.float32) - DS_factor_track//2 + 0.5 ##Unity slope track
        stencilT /= DS_factor_track
    
        stencilS = np.arange(DS_factor_scan,  dtype=np.float32) - DS_factor_scan//2 + 0.5 ##Unity slope scan
        stencilS = stencilS.reshape((2,DS_factor_scan//2)) / DS_factor_scan 
    
        fOut = DS[:,x,:,x,x] + stencilT[x,:,x,x,x]*(slopeT[:,x,:,x,x] + stencilS[x,x,x,:,:]*slopeST[:,x,:,:,x]) + stencilS[x,x,x,:,:]*slopeS[:,x,:,:,x] 
    
        return fOut.reshape((nT,nS))
    
    #=========================================================================================
    
    def getVectors(self,xres,yres,zres, lps, segs):
        ## Get unit vectors along and across the scan
        ## to get/apply component corrections
        
        ### lps = lines_per_scan
        nT,nS = xres.shape
    
        midscan=slice(lps//2,nT,lps)
        seg1=slice(0,    nS,segs) 
        seg2=slice(segs-1,nS,segs) 
    
        x1,y1,z1 = xres[midscan, seg1],   yres[midscan, seg1],   zres[midscan, seg1]
        x2,y2,z2 = xres[midscan, seg2],   yres[midscan, seg2],   zres[midscan, seg2]
        
        ## Unity vector along the scan
        vecnorm = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) 
        if np.any(vecnorm < 1):
            vecnorm[vecnorm < 1] = np.nan
            #Should not happen if the missing values are encoded properly
            #raise ValueError("Failed getVectors on scan norm")
    
        esx = (x2 - x1) / vecnorm
        esy = (y2 - y1) / vecnorm
        esz = (z2 - z1) / vecnorm
    
    
        ## Norm to the scan plane for each scan
        xn  =   y1*z2 - z1*y2
        yn  =   z1*x2 - x1*z2
        zn  =   x1*y2 - y1*x2
        tracknorm = np.sqrt(xn**2 + yn**2 + zn**2)
        if np.any(tracknorm < 1):
            tracknorm[tracknorm < 1] = np.nan

        etx = xn/tracknorm
        ety = yn/tracknorm
        etz = zn/tracknorm
    
        return esx,esy,esz,etx,ety,etz
    
    #==============================================================================
    
    def CorrST(self,xres,yres,zres, x, y, z, lps, segs) :
        #
        # Returns correction in along-scan and along-track directions
        #
        nT,nS = xres.shape
        byscan=((nT//lps,lps,nS//segs,segs))
    
    #    print(esx*etx + esy*ety + esz*etz)
    
        deltax = (x - xres).reshape(byscan)
        deltay = (y - yres).reshape(byscan)
        deltaz = (z - zres).reshape(byscan)
    
        esx,esy,esz,etx,ety,etz = self.getVectors(xres,yres,zres, lps, segs)
    
        corrs = deltax * esx[:,np.newaxis,: ,np.newaxis ] + deltay * esy[:,np.newaxis,: ,np.newaxis ] + deltaz * esz[:,np.newaxis,: ,np.newaxis ]
        corrt = deltax * etx[:,np.newaxis,: ,np.newaxis ] + deltay * ety[:,np.newaxis,: ,np.newaxis ] + deltaz * etz[:,np.newaxis,: ,np.newaxis ]
    
        if False:  ## Verbose output of correction magnitudes
            prcval = [0, 1,  5, 50, 95, 99, 99.9, 100]
            percentiles = np.percentile(corrt**2, prcval)
            print( "corrt percentiles: %s", ", ".join([ "prc%d = %4.0fm"%(v,np.sqrt(p)) for v,p in zip(prcval,percentiles)]))
            percentiles = np.percentile(corrs**2, prcval)
            print( "corrs percentiles: %s", ", ".join([ "prc%d = %4.0fm"%(v,np.sqrt(p)) for v,p in zip(prcval,percentiles)]))
    
        return corrs.reshape((nT,nS)), corrt.reshape((nT,nS))
     
    #===================================================================================   
    
    def ApplyCorrST(self,xres,yres,zres, corrs, corrt, lps, segs) :
        #
        # Applies correction in along-scan and along-track directions to "restored" values
        #
        nT,nS = xres.shape
        byscan=((nT//lps,lps,nS//segs,segs)) 
        
        xa = xres.reshape(byscan)
        ya = yres.reshape(byscan)
        za = zres.reshape(byscan)
    
        esx,esy,esz,etx,ety,etz = self.getVectors(xres,yres,zres, lps, segs)
    
        scs = corrs.reshape(byscan)
        sct = corrt.reshape(byscan)
    
        x = xa + scs * esx[:,np.newaxis, : ,np.newaxis ] + sct * etx[:,np.newaxis, : ,np.newaxis ]
        y = ya + scs * esy[:,np.newaxis, : ,np.newaxis ] + sct * ety[:,np.newaxis, : ,np.newaxis ]
        z = za + scs * esz[:,np.newaxis, : ,np.newaxis ] + sct * etz[:,np.newaxis, : ,np.newaxis ]
    
        return x.reshape((nT,nS)), y.reshape((nT,nS)), z.reshape((nT,nS)),
    
    #===========================================================================
    
    def ApplyCorrLonLatTan(self, x_restore,y_restore,z_restore, dr_phi, dr_theta) :
    
        #
        # Applies correction in lon-lat tangent 
        #
        phi, theta = gridtools.phithetaFromXYZ(x_restore, y_restore, z_restore)
        xr = x_restore  + (-  dr_phi * np.sin(phi) - dr_theta * np.sin(theta) * np.cos(phi))
        yr = y_restore  + (   dr_phi * np.cos(phi) - dr_theta * np.sin(theta) * np.sin(phi))
        zr = z_restore  +                            dr_theta * np.cos(theta)
    
        return xr, yr, zr
    
    #===========================================================================
    
    def trimLonLat(self,lon, lat, dx_m):
            ### Assumes that lon and lat come in degrees
            ### dx_m  -- quantum in meters
            ##  FIXME For some reason does not improve size
    
        log2quantum = int(np.floor(np.log2(dx_m/111000.)))  - 1  ## Quantum for lon-lat 
                                                    ## less than half of dx_m at equator
        if False:
            latprev = 0
            rad2deg = 180./np.pi
            for q in range(log2quantum,1):
                latbnd = rad2deg * np.arcsin(1. - 2**(log2quantum - q - 1)) ##Belt where this precision applies
                idx = np.where((abs(lat)>=latprev)*(abs(lat)<latbnd))
                util.trimPrecisionAbs(lon[(abs(lat)>=latprev)*(abs(lat)<latbnd)], q - 2)
                print (latprev,latbnd, q + 2, len(idx[0]))
                latprev = latbnd
                if latbnd > 85: 
                    break
            util.trimPrecisionAbs(lon[(abs(lat)>=latprev)], q) ## Trim last longitude
        else:
            util.trimPrecisionAbs(lon, log2quantum)
        util.trimPrecisionAbs(lat, log2quantum)
    
    #===========================================================================
    
    def upsample_lonlat_toxyz(self,lon_DS, lat_DS,DS_factor_track, DS_factor_scan):
        x_DS, y_DS, z_DS = gridtools.xyzFromLonLat(lon_DS, lat_DS)
        x_restore = self.upsample_fld(x_DS, self.scanSz_375, self.idx_segments,  DS_factor_track, DS_factor_scan)
        y_restore = self.upsample_fld(y_DS, self.scanSz_375, self.idx_segments,  DS_factor_track, DS_factor_scan)
        z_restore = self.upsample_fld(z_DS, self.scanSz_375, self.idx_segments,  DS_factor_track, DS_factor_scan)
        return x_restore, y_restore, z_restore
    
    #=================================================================================    
    
    def getLonLatFromFile(self, inf, file_type=None):
    
        with  nc4.Dataset(inf) as geof:
            geof.set_auto_mask(False) ## No mask -- faster
            # global attributes
            dicGlobal_attr = {}
            for a in geof.ncattrs():
                dicGlobal_attr[a] = geof.getncattr(a)
            # groups and group attributes
            for g in geof.groups.keys():
                for a in geof[g].ncattrs():
#                    key = "%s_%s"%(g,a)
                    dicGlobal_attr[a] = geof[g].getncattr(a)
    
            self.check_file_type(geof.ShortName)  # just to make sure that we know how to handle it
    
            if 'file_type' in dicGlobal_attr:
                if file_type is None:
                    granule_type = dicGlobal_attr['file_type']
                else:
                    granule_type = file_type
                    print("Forcing file_type to ", file_type)
                    
                if granule_type in  'downsampled_scantrack downsampled_lonlat_tangent downsampled_lonlat_2'.split():
    
                    lon_DS = geof.variables['longitude'][:]
                    lat_DS = geof.variables['latitude'][:]
                    DS_factor_track = geof.getncattr('downsampling_factor_along_track')    # available only for downsampled set
                    DS_factor_scan = geof.getncattr('downsampling_factor_along_scan')    # available only for downsampled set
                    x_restore, y_restore, z_restore = self.upsample_lonlat_toxyz(
                                                    lon_DS, lat_DS, DS_factor_track, DS_factor_scan)
    
                    if granule_type == 'downsampled_scantrack':
                        try:
                                scantracksegment = dicGlobal_attr['downsampled_scantrack_segment_size']
                        except KeyError:
                                scantracksegment = 128  ## Older files have no such attribute 
                                dicGlobal_attr['downsampled_scantrack_segment_size'] = scantracksegment
                        dscan  = geof.variables['dscan'][:]
                        dtrack = geof.variables['dtrack'][:]
                        x, y, z =  self.ApplyCorrST(x_restore,y_restore,z_restore, dscan, dtrack, 
                                                    self.scanSz_375, scantracksegment)
                        lon, lat =  gridtools.lonlatFromXYZ(x, y, z)
    
                    elif granule_type ==  'downsampled_lonlat_2': ## MAS files
                        dlonm = geof.variables['dlon_int'][:].astype(np.float32)
                        dlatm = geof.variables['dlat_int'][:].astype(np.float32)
                        unit = geof.variables['dlon_int'].units
                        if unit[-1] == 'm'  and unit != 'm':
                            factor = float(unit[0:-1]) ## Handle old MAS units like "40m"
                            dlonm *= factor
                            dlatm *= factor
                        lon_restore, lat_restore = gridtools.lonlatFromXYZ(x_restore, y_restore, z_restore)
    
                        dlondeg = dlonm / (111e3 * np.cos(lat_restore * np.pi / 180. ))
                        dlatdeg = dlatm /(111e3)
                        ## Back-forth to handle wraps
                        x, y, z = gridtools.xyzFromLonLat(lon_restore - dlondeg , lat_restore - dlatdeg)
    
                    elif granule_type ==  'downsampled_lonlat_tangent':
                        dr_phi   = geof.variables['dr_phi_int'][:].astype(np.float32)
                        dr_theta = geof.variables['dr_theta_int'][:].astype(np.float32)
                        assert (geof.variables['dr_phi_int'].units == 'm')
                        phi, theta = gridtools.phithetaFromXYZ(x_restore, y_restore, z_restore)
                        x = x_restore  + (-  dr_phi * np.sin(phi) - dr_theta * np.sin(theta) * np.cos(phi))
                        y = y_restore  + (   dr_phi * np.cos(phi) - dr_theta * np.sin(theta) * np.sin(phi))
                        z = z_restore  +                                dr_theta * np.cos(theta)
                    else:
                        raise ValueError("Should never get here!")
    
                    lon, lat =  gridtools.lonlatFromXYZ(x, y, z)
                else:
                    raise ValueError("Unknown granule_type = %s"%(granule_type))
    
            else: 
                if  'HDFEOS' in geof.groups:  #(VNP03IMGLL, coll 5000) 
                    fInInt = geof['HDFEOS']['SWATHS']
                    lon = fInInt['VNP_375M_GEOLOCATION']['Geolocation Fields']['Longitude'][:,:]
                    lat = fInInt['VNP_375M_GEOLOCATION']['Geolocation Fields']['Latitude'][:,:]
                elif hasattr(geof, 'title') and geof.title == "VIIRS I-band Geolocation Data": ### VNP03IMG, VJ103IMG, (coll 5200)
                    fInInt = geof.groups['geolocation_data'].variables
                    lon = fInInt['longitude'][:,:]
                    lat = fInInt['latitude'][:,:]
                else:
                    raise AttributeError("Could not find geolocation from %s"%(inf,))
    
        return lon, lat, dicGlobal_attr
    
    #=======================================================================================
    
    def downsample_field(self,DS_factor_track, DS_factor_scan, scanSz, fldIn, method='mid4'):
        #
        #
        if DS_factor_track == 1 and DS_factor_scan == 1: return fldIn
        if not (DS_factor_track, DS_factor_scan) in [(2,2),(4,4),(8,8),(16,16),(16,32)]:
            raise ValueError('Downsample factor must be a power of 2')
        #
        nTrack, nScan = fldIn.shape
    
        #Make downsampled pieces to separate dimensions 
        fIn = fldIn.reshape((nTrack // scanSz, scanSz//DS_factor_track, DS_factor_track, nScan//DS_factor_scan, DS_factor_scan))  # (202,32/n,n,6400,n,n)
    
        # New field (nScans, pixels_in_scan_along_track, pixels_along_scan)
        if method == 'mid4': 
            ## Faster quivalent to the MAS method
            sls = slice(DS_factor_scan//2-1,DS_factor_scan//2+1)
            slt = slice(DS_factor_track//2-1,DS_factor_track//2+1)
            f_av = np.mean(fIn[:,:,slt,:,sls], axis=(2,4))
        elif method == 'median': 
            ## marginally better compression (1-2%), a some 5% more expensive
            f_av = np.mean(fIn[:,:,:,:,:], axis=(2,4))
        else:
            raise ValueError("Unknown method (\"%s\") can be one of mid4 or median"%(method))
        
        return f_av.reshape(f_av.shape[0] * f_av.shape[1], f_av.shape[2])
    
    #============================================================================================
    
    def DumpToFile(self,lon, lat, dicGlobal_attr_in, 
                   iFactor_track, iFactor_scan, correction_sensitivity_m, chFNmOut):
    
        dicGlobal_attr = dicGlobal_attr_in.copy()
    
#        geom = getGeometryByShortName(dicGlobal_attr["ShortName"])
    
        nTrack,nScan = lon.shape
        if  self.scanSz_375 % iFactor_track != 0:
            raise ValueError("geom.scanSz % iFactor_track != 0") ## 2 points per scan
        if nScan % iFactor_scan != 0:
            raise ValueError("Downsample factor (%d) is not a fraction of scan size (%d)"%(iFactor_scan, nScan))
    
        try:
            scantracksegment = dicGlobal_attr['downsampled_scantrack_segment_size']
        except KeyError:
            scantracksegment = 128
            dicGlobal_attr['downsampled_scantrack_segment_size'] = scantracksegment
    
    
        x, y, z = gridtools.xyzFromLonLat(lon, lat)
        method = 'mid4'
        x_DS = self.downsample_field(iFactor_track, iFactor_scan, self.scanSz_375, x, method)
        y_DS = self.downsample_field(iFactor_track, iFactor_scan, self.scanSz_375, y, method)
        z_DS = self.downsample_field(iFactor_track, iFactor_scan, self.scanSz_375, z, method)
    
        dicGlobal_attr['downsampling_factor_along_track'] = iFactor_track
        dicGlobal_attr['downsampling_factor_along_scan']  = iFactor_scan
    
        if not "file_type" in dicGlobal_attr:
            dicGlobal_attr['file_type'] = 'downsampled_scantrack'
    
        ## Trimmed-precision lon-lat to store
    
        lon_DS, lat_DS = gridtools.lonlatFromXYZ(x_DS, y_DS, z_DS)
    
        log2quantum = int(np.floor(np.log2(correction_sensitivity_m/1.1e5))+1.) - 2 
        ## 13.5 m quantum for correction_sensitivity_m = 40 -- apparently best compression
        dicGlobal_attr['log2quantumLon'] = log2quantum
        dicGlobal_attr['log2quantumLat'] = log2quantum
    
        util.trimPrecisionAbs(lon_DS, log2quantum)
        util.trimPrecisionAbs(lat_DS, log2quantum)
        ##Restore the downsampled xyz from stored trim-precision values
        x_DS, y_DS, z_DS = gridtools.xyzFromLonLat(lon_DS, lat_DS)
    
        xres, yres, zres = self.upsample_lonlat_toxyz(lon_DS, lat_DS, iFactor_track, iFactor_scan)
    
    
        corrs, corrt = self.CorrST(xres, yres, zres, x, y, z, self.scanSz_375, scantracksegment)
    
        self.writeSTdownsampled(chFNmOut, lon_DS, lat_DS, corrs, corrt, dicGlobal_attr, 
                                correction_sensitivity_m)
    
    #==============================================================================================================
    
    def writeSTdownsampled(self,chFNmOut, lon_DS, lat_DS, corrs, corrt, dicGlobal_attr, 
                           correction_sensitivity_m):
        #
        # Dump downsampled granule to file
        
        dirname = os.path.split(chFNmOut)[0]
    
        if dirname != '' and  not os.path.exists(dirname):
            os.makedirs(dirname)
        # write file
        chFNmOutTmp = chFNmOut + '_tmp.pid%08d'%(os.getpid(),)
    
        with nc4.Dataset( chFNmOutTmp, "w", format="NETCDF4") as outf:
    
            nTrack, nScan = corrs.shape
            outf.createDimension("Along_Track", nTrack)
            outf.createDimension("Along_Scan", nScan)
            valdims_full = ("Along_Track","Along_Scan",)
            
            nT_DS, nS_DS = lon_DS.shape
            outf.createDimension("Along_Track_DS", nT_DS)
            outf.createDimension("Along_Scan_DS", nS_DS)
            valdims_DS = ("Along_Track_DS","Along_Scan_DS",)
            arStore = [("longitude","longitude","f4","degrees_east", valdims_DS, lon_DS),
                       ("latitude","latitude",'f4',"degrees_north", valdims_DS, lat_DS),
                       ("dscan", "dscan", 'i2',"m", valdims_full, corrs),
                       ("dtrack","dtrack",'i2',"m", valdims_full, corrt)]
    
            # Store the stuff
            for Nm, NmLong, tp, u, valdims, v in arStore:
                var = outf.createVariable(Nm, tp, valdims, zlib=True, complevel=9)
                var.set_auto_maskandscale(False) ##Disable creativity of the netcdf4 library
                                            ## We can do it way better
                var.standard_name = Nm
                var.long_name = NmLong
                var.units = u
                if tp == 'i2' and u == 'm':# Pack the thing using CF packing
                    var.scale_factor = correction_sensitivity_m ##Get scale
                    maxval = np.amax(np.abs(v) / var.scale_factor)
                    if maxval < 32768 - 128: ## Can use ine byte out of two for near-zero values: better compression
                        var.add_offset = - 128 * var.scale_factor
                        var[:,:] = np.rint(v/var.scale_factor).astype(dtype=np.int16) + 128
                    else:
                        raise ValueError("Variable range can't be packed to int16 with given scale factor")
                else:
                    var[:,:] = v
            #
            # Write metadata: groups and attributes of the oroginal file            
            for k,v in dicGlobal_attr.items():
                key = k.replace('/','_').replace(' ','_')  #Slash is not allowed in attr name, space just for compatibility with MAS tools
                outf.setncattr(key,v)
    
        os.replace(chFNmOutTmp, chFNmOut)
    
    #==========================================================================================
    
    def evaluateErr(self, lon, lat, lon1,lat1, label):
    
        x , y , z  = gridtools.xyzFromLonLat(lon ,lat )
        x1, y1, z1 = gridtools.xyzFromLonLat(lon1,lat1)
    
    
        #### report error percentiles
        prcval = [0, 5, 50, 95, 100]
    
        err2 =  (x-x1)**2 + (y-y1)**2 + (z-z1)**2 
        percentiles = np.percentile(err2, prcval) ##percentiles of _squared_ error
        print( "%s: RMSE = %8.3fm, Error percentiles: %s"% 
                    ( label, np.sqrt(np.mean(err2)), 
                      ", ".join([ "prc%d = %4.0fm"%(v,np.sqrt(p)) for v,p in zip(prcval,percentiles)])))
    
    
        return  np.sqrt(percentiles[-1])
    
    #=============================================================

    def pick_granule_data_IS4FIRES_v3_0(self, fIn_fire=None, fIn_aux=None, grid_to_fit_in=None, ifHeightCorrection=False):
        #
        # Following the above paradigm, we need:
        # - from VNP/V?? 14:
        #      - FRP for fire pixels, their temperatures, locations in swath and lon-lat
        #      - cloud mask, land, water, cloud, desert, day/night, QA
        # - from VNP/V?? 03:
        #      - longitude, latitude, height for geolocation
        #
        # Note the difference form MODIS: there, MxD35 contains the bitmask, whereas
        # here it is in the 14 product together with the main fire data.
        #
        # The reader supports two options: reading by default settings, i.e., self.now and 
        # file name templates, and by reading the given files - then time etc is taken from 
        # the files and updated in the self.
        #
        # Start from geolocation
        #
        if fIn_aux is not None:
            self.chFNm_geo = fIn_aux
        else:
            if self.now_UTC is None:   # a single file must be given, time will be taken from it
                arFNms = [self.templAuxilFiles]
            else:
                arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templAuxilFiles)))
            if len(arFNms) == 1:
                if not self.ifSilent: 
                    self.log.log('Reading geolocation from ' + os.path.split(arFNms[0])[1] + 
                                 dt.datetime.now().strftime(', time: %H:%M.%S'))
                self.chFNm_geo = arFNms[0]
            elif len(arFNms) > 1:
                if not self.ifSilent: 
                    self.log.log('Several geolocation files satisfy template:' + str(arFNms))
                self.chFNm_geo = arFNms[-1]
    #            return False
            else:
                if not self.ifSilent: 
                    self.log.log(self.now_UTC.strftime('%Y%m%d_%H%M: No auxiliary files for template:') + 
                                 self.templAuxilFiles)
                return False
        #
        # read-and-unpack, do not forget to reset time in self
        #
        self.lon, self.lat, self.dicGlobal_attr = self.getLonLatFromFile(self.chFNm_geo)
        self.now_UTC = dt.datetime.strptime(self.dicGlobal_attr['StartTime'], '%Y-%m-%d %H:%M:%S.000')   # "2020-01-01 01:36:00.000"
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
        # Geolocation is known. Get FRP
        # FRP file has two parts: Fire-Pixel Table and packed maps - another set of bytes
        # Same as before, can be a default reading via self.now and tempaltes, or just a given name
        #
        if fIn_fire is not None:
            self.chFNm_FRP = fIn_fire
        else:
            arFNms = sorted(glob.glob(self.now_UTC.strftime(self.templFRPFiles)))
            if len(arFNms) == 1:
                if not self.ifSilent: 
                    print('Reading FRP from ', os.path.split(arFNms[0])[1])
                self.chFNm_FRP = arFNms[0]
            elif len(arFNms) > 1:
                if not self.ifSilent: 
                    self.log.log('Several FRP files satisfy template:' + str(arFNms))
                self.chFNm_FRP = arFNms[-1]
    #            return False
            else:
                if not self.ifSilent: 
                    self.log.log('Template: ' + self.templFRPFiles)
                    self.log.log('Found files: ' + '\n'.join(glob.glob(self.now_UTC.strftime(self.templFRPFiles))))
                    self.log.log('Something went wrong: no files for template')
                return False
        # open
        fIn = nc4.Dataset(self.chFNm_FRP, 'r')
        fIn.set_auto_mask(False) ## No mask -- faster
        dateFire = dt.datetime.strptime(fIn.getncattr('StartTime') , '%Y-%m-%d %H:%M:%S.000')
        if dateFire != self.now_UTC:
            raise AttributeError('Time of geolocation and fire files are not the same: %s %s' % 
                                 (self.now_UTC.strftime('%Y%m%d-%H%M'), fIn.getncattr('StartTime')))
        #
        # Get the cloud mask from v??_14
        # direct way fails, use this way - checked to work
        #
        cld_packed = fIn['fire mask'][:,:]
#        cld_bytes = np.zeros(shape=cld_packed.shape, dtype=np.uint8)  #'uint8')
#        cld_bytes[:,:,:] = cld_packed[:,:,:]
#        bits = np.unpackbits(cld_bytes,axis=0)
        self.BitFields = self.unpack_Vx_QA(cld_packed) #, fIn.DayNightFlag == 'Day')
        self.BitFields.day_night = np.where(spp.solar_zenith_angle(self.lon, self.lat, 
                                                                   self.now_UTC.timetuple().tm_yday,
                                                                   self.now_UTC.hour, self.now_UTC.minute) > 90.,
                                            0, 1)   # same as in MODIS:  0 = Night  / 1 = Day
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
        # ATTENTION.
        # Channels I5 corresponds to temperature at 11 microns, so name will be FP_11 and FP_11b
        #
        if fIn.FirePix > 0:
            #
            # Rest will continue with the fire records cless
            #
            self.nFires = fIn.FirePix
            self.FP = fire_records.fire_records(self.log)
            self.FP.nFires = self.nFires
            self.FP.init_data_structures(self.nFires)
            self.FP.FRP = fIn.variables['FP_power'][:]
            self.FP.line = fIn.variables['FP_line'][:]     # probably index along the track [0:6432]
            self.FP.sample = fIn.variables['FP_sample'][:] # probably, along the scan, [0:6400]
            self.FP.lon = fIn.variables['FP_longitude'][:]
            self.FP.lat = fIn.variables['FP_latitude'][:]
#            print('min-max of lines along line and across sample',
#                  np.min(self.FP_line), np.max(self.FP_line), 
#                 np.min(self.FP_sample), np.max(self.FP_sample))
            self.FP.dS = self.dS[self.FP.sample]   # depends on swath 
            self.FP.dT = self.dT[self.FP.sample]   # also depends on swath
            self.FP.T4 = fIn.variables['FP_T4'][:]
            self.FP.T4b = fIn.variables['FP_MeanT4'][:]
            self.FP.T11 = fIn.variables['FP_T5'][:]        # channel I5 corresponds to 11 um
            self.FP.T11b = fIn.variables['FP_MeanT5'][:]   # channel I5 corresponds to 11 um
            self.FP.TA = fIn.variables['FP_MeanDT'][:]
            self.FP.SolZenAng = fIn.variables['FP_SolZenAng'][:]
            self.FP.ViewZenAng = fIn.variables['FP_ViewZenAng'][:]
            self.FP.satellite = np.array([self.abbrev_1(self.chFNm_FRP)]*fIn.FirePix, dtype='|S1')
            self.FP.time = np.zeros(shape=(self.nFires),dtype=np.int64)
            self.FP.QA_flag = 0
            self.FP.QA_msg = ''
            self.FP.timezone = 'UTC'
            self.FP.LU_metadata = ''
            self.FP.timeStart = self.now_UTC
            if not self.ifSilent: 
                self.log.log('Nbr of fires, FRP: %g, %g MW, %s'  % 
                             (self.nFires, np.sum(self.FP.FRP), str(self.now_UTC)))
        else:
            # No fires
            if not self.ifSilent: self.log.log('No fires')
            self.nFires = 0
        return True


    #=============================================================

    def unpack_Vx_QA(self, arIn): # , dayNight):
        #
        # This sub mimics the MODIS QA legend by translating the VIIRS AQ bits into MODIS
        # MODIS uses bitwise packing while VIIRS just codes the value
        #
        # VIIRS QA meaning:
        # 0 not-processed (non-zero QF)     1 bowtie
        # 2 glint                           3 water
        # 4 clouds                          5 clear land
        # 6 unclassified fire pixel         7 low confidence fire pixel
        # 8 nominal confidence fire pixel   9 high confidence fire pixel
        #
        # MODIS QA bits:
        # Bit fields within each byte are numbered from the left:
        # 7, 6, 5, 4, 3, 2, 1, 0.
        # The left-most bit (bit 7) is the most significant bit.
        # The right-most bit (bit 0) is the least significant bit.
        #
        # bit field       Description                             Key
        # ---------       -----------                             ---
        # 0               Cloud Mask Flag                      0 = Not  determined
        #                                                      1 = Determined
        # 2, 1            Unobstructed FOV Quality Flag        00 = Cloudy
        #                                                      01 = Uncertain
        #                                                      10 = Probably  Clear
        #                                                      11 = Confident  Clear
        #                 PROCESSING PATH
        #                 ---------------
        # 3               Day or Night Path                    0 = Night  / 1 = Day
        # 4               Sunglint Path                        0 = Yes    / 1 = No
        # 5               Snow/Ice Background Path             0 = Yes    / 1 = No
        # 7, 6            Land or Water Path                   00 = Water
        #                                                      01 = Coastal
        #                                                      10 = Desert
        #                                                      11 = Land
        #
        granBits = gb.unpacked_cloud_fields()  # Create the object for storage
        granBits.ifAnalysed = arIn > 2      # processed, not bowtie, not glint
        granBits.QA = arIn * 0         # everything is cloud
#        granBits.QA[arIn == 4] = 0          # cloudy
        granBits.QA[arIn == 3] = 11         # water, clearsky
        granBits.QA[arIn >= 5] = 11         # clearsky land, fires of all kinds 
#        granBits.day_night = dayNight
        granBits.sunglint = arIn != 2   # sunglint => 1
#        granBits.sunglint[np.logical_or(arIn > 4, arIn == 3)] = 0
        granBits.snow = 0          # not analysed by VIIRS
        granBits.land = arIn * 0     # everything is water
        granBits.land [arIn >= 5] = 11  # land and all fire pixels
        granBits.land [arIn == 3] = 0   # just to make sure that water is water
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
        outf = nc4.Dataset(chFNmOut + '_tmp', "w", format="NETCDF4")
        outf.featureType = "VIIRS_geo";
        # store the type
        if ifDownsampled:
            typeTmp = self.granule_type
        else:
            typeTmp = 'original'
        outf.setncattr('file_type',typeTmp)
        # array shapes: full and, if needed, downsampled
        shp_full = self.lon.shape
        outf.createDimension("Along_Track", shp_full[0])
        outf.createDimension("Along_Scan", shp_full[1])
        valdims_full = ("Along_Track","Along_Scan",)
        if 'downsampled_lonlat' in typeTmp: 
            shp_DS = self.lon_DS.shape
        elif 'downsampled_cartesian' in typeTmp: 
            shp_DS = self.x_DS.shape
        elif typeTmp == 'original' or typeTmp == 'external': 
            shp_DS = None
        else: 
            raise ValueError('Unknown type of granule:' + typeTmp)
        # Dimensions
        if not shp_DS is None:
            outf.createDimension("Along_Track_DS", shp_DS[0])
            outf.createDimension("Along_Scan_DS", shp_DS[1])
            valdims_DS = ("Along_Track_DS","Along_Scan_DS",)
        # What to store:
        if typeTmp == 'downsampled_lonlat':
            arStore = [("longitude","longitude","f4","degrees_east", 1, valdims_DS, self.lon_DS),
                       ("latitude","latitude",'f4',"degrees_north", 1, valdims_DS, self.lat_DS),
                       ("dx_int","dx-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dx_int),
                       ("dy_int","dy-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dy_int),
                       ("dz_int","dz-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dz_int)]
            if not self.height_DS is None:
                arStore.append(('height',"altitude above sea level",'i2','meters', 1, valdims_DS, self.height_DS))
        elif typeTmp == 'downsampled_cartesian':
            arStore = [("x","x-Cartezian","f4","km", 1, valdims_DS, self.x_DS),
                       ("y","y-Cartesian",'f4',"km", 1, valdims_DS, self.y_DS),
                       ("z","z-Cartesian",'f4',"km", 1, valdims_DS, self.z_DS),
                       ("dx_int","dx-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dx_int),
                       ("dy_int","dy-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dy_int),
                       ("dz_int","dz-Cartesian",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dz_int)]
            if not self.height_DS is None:
                arStore.append(('height',"altitude above sea level",'i2','meters', 1, valdims_DS, self.height_DS))
        elif typeTmp == 'downsampled_lonlat_2':
            arStore = [("longitude","longitude","f4","degrees_east", 1, valdims_DS, self.lon_DS),
                       ("latitude","latitude",'f4',"degrees_north", 1, valdims_DS, self.lat_DS),
                       ("dlon_int","dlon",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dlon_int),
                       ("dlat_int","dlat",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dlat_int)]
            if not self.height_DS is None:
                arStore.append(('height',"altitude above sea level",'i2','meters', 1, valdims_DS, self.height_DS))
        elif typeTmp == 'downsampled_cartesian_2':
            arStore = [("x","x-Cartezian","f4","km", 1, valdims_DS, self.x_DS),
                       ("y","y-Cartesian",'f4',"km", 1, valdims_DS, self.y_DS),
                       ("z","z-Cartesian",'f4',"km", 1, valdims_DS, self.z_DS),
                       ("dlon_int","dlon",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dlon_int),
                       ("dlat_int","dlat",'i2',"%im" % self.corr_scale, 1, valdims_full, self.dlat_int)]
            if not self.height_DS is None:
                arStore.append(('height',"altitude above sea level",'i2','meters', 1, valdims_DS, self.height_DS))
        elif typeTmp == 'original' or typeTmp == 'external':
            arStore = [("longitude","longitude","f4","degrees_east", 1, valdims_full, self.lon),
                       ("latitude","latitude",'f4',"degrees_north", 1, valdims_full, self.lat)]
            if not self.height is None:
                arStore.append(('height',"altitude above sea level",'i2','meters', 1, valdims_full,self.height))
        else: raise ValueError('Unknown type of granule:' + typeTmp)
        # Store the stuff
        for Nm, NmLong, tp, u, sf, valdims, v in arStore:
            var = outf.createVariable(Nm, tp, valdims, zlib=True, complevel=9, shuffle=True)
            var.standard_name = Nm
            var.long_name = NmLong
            var.units = u
            var.scale_factor = sf
            var[:,:] = v
        #
        # Write metadata: groups and attributes of the original file            
        for a in self.dicGlobal_attr.keys():
            if type(self.dicGlobal_attr[a]) == 'str':
                outf.setncattr(a.replace(' ','_').replace('/','_'), 
                               self.dicGlobal_attr[a].replace(' ','_').replace('/','_'))
            else:
                outf.setncattr(a.replace(' ','_').replace('/','_'), self.dicGlobal_attr[a])
        #
        # Mark downsampled field
        if ifDownsampled:
            outf.setncattr('downsampling_factor_along_track', self.DS_factor_track)
            outf.setncattr('downsampling_factor_along_scan', self.DS_factor_scan)
        #
        # groups and group attributes
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
        outf.close()
        os.replace(chFNmOut + '_tmp', chFNmOut)
#        if not self.ifSilent: print('Stored without renaming:', chFNmOut)


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
        FP_frp = fIn.variables['FP_power'][:]
        self.nFires = len(FP_frp)
        if self.nFires == 0: return True         # empty granule
        # fire_records further on
        self.FP = fire_records.fire_records(self.log)
        self.FP.init_data_structures(self, self.nFires)
        self.FP.FRP = fIn.variables['FP_power'][:]
        self.FP.line = fIn.variables['FP_line'][:]     # index along the track [0:2029]
        self.FP.sample = fIn.variables['FP_sample'][:] # along the scan, [0:1353]
        self.FP.lon = fIn.variables['FP_longitude'][:]
        self.FP.lat = fIn.variables['FP_latitude'][:]
        if not self.ifSilent: 
            print(np.min(self.FP.line), np.max(self.FP.line), 
                  np.min(self.FP.sample), np.max(self.FP.sample))
        self.FP.dS = self.dS[self.FP.sample]
        self.FP.dT = self.dT[self.FP.sample]
        self.FP.T4 = fIn.variables['FP_T21'][:]
        self.FP.T4b = fIn.variables['FP_MeanT21'][:]
        self.FP.T11 = fIn.variables['FP_T31'][:]
        self.FP.T11b = fIn.variables['FP_MeanT31'][:]
        self.FP.TA = fIn.variables['FP_MeanDT'][:]
        self.FP.SolZenAng = fIn.variables['FP_SolZenAng']
        self.FP.satellite = np.array([self.abbrev_1(arFNms[0])] * self.nFires, dtype='|S1')
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
        for iFire in range(self.FP_frp.shape[0]):
            fOut.write('fire = %03i %s %g %g %g %g km %g MW %g %g %g %g %g %g %g %g %g\n' %
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

        for var, chTxt, cm in [(self.BitFields.ifAnalysed,'If analysed (1=yes) + FRP, [MW]','winter'),
                               (self.BitFields.QA,'QA: 0=cld,1=?,10=clr?,11=clr','Paired'),
                               (np.ones(shape=self.lon.shape) * self.BitFields.day_night, 
                                '1=day, 0=night','cool'), 
                               (self.BitFields.sunglint,'1=sunglint','cool'),
#                               (np.ones(shape=lon.shape) * self.BitFields.snow,'0=snow','cool'),
#                               (self.height,'height, m','terrain'),
                               (np.repeat(self.area[None,:], self.lon.shape[0], axis=0),'area, km2','cool'),
                               (self.BitFields.land,'0=water,1=coast,10=desert,11=land','Paired'),
                               (self.lon,'Lon','cool'),
                               (self.lat,'Lat','cool')]:
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
                chFires = ', %g fires' % self.FP.FRP.shape[0]
                sort_order = self.FP.FRP.argsort()
                cs2 = bmap.scatter(self.FP.lon[sort_order], self.FP.lat[sort_order], 
                                   c=self.FP.FRP[sort_order], s=30, 
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
        mpl.pyplot.savefig(chOutFNm,dpi=200)
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
                sort_order = self.FP.FRP.argsort()
                cs2 = bmap.scatter(self.FP.lon[sort_order], self.FP.lat[sort_order], 
                                   c=self.FP.FRP[sort_order], s=30, 
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

def downsample_geo_granule(chFNm_geo375, chFNm_FRP, chFNmHeight, now, iFactor_track, iFactor_scan,
                           correction_sensitivity_m, chDirOut, ifHeightCorrection, ifVerify, log):
    #
    # Downsamples a single time moment
    # Procedure is the following:
    # - do the downsampling step
    # - do the up-sampling step
    # - compare the down-up-sampled map and get the adjustment maps
    # - write the stuff to the file
    #
    timer_start = dt.datetime.now()
    print('DS start, time=', timer_start.strftime('%Y-%m-%d %H:%M'), 'granule:', 
          now.strftime('%Y-%m-%d %H:%M'), chFNm_FRP)
    gv_375 = granule_VIIRS('VIIRS', now, chFNm_FRP, chFNm_geo375, chFNmHeight, log)
#                           spp.log(os.path.join(chDirLog,now.strftime('log_%Y%m%d_%H%M.log'))))
    if not gv_375.pick_granule_data_IS4FIRES_v3_0(ifHeightCorrection): return -1
    #
    # downsample the granule in Cartesian coordinates.
    # Note that it also creates correction arrays
    #
    chFNmOut = os.path.join(chDirOut, #'DS_%02i_%02i' % (iFactor_track,iFactor_scan),
#                            gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
                            gv_375.now().strftime('%Y.%m.%d'),
                            os.path.split(gv_375.get_geo_FNm())[1] + '.nc4')
    gv_375.downsample_granule_geo_Cartesian_2(iFactor_track, iFactor_scan, 
                                            True, ifHeightCorrection, ifHeightCorrection, # ifReturnLonLat, ifHeightToo, ifHeightCorrection
                                            correction_sensitivity_m, chFNmOut)
    print('Done downsampling. Time used:', dt.datetime.now() - timer_start)
    #
    # If verification needed, do it here
    #
    if ifVerify:
        timer_start_verify = dt.datetime.now()
        gv_375_r = granule_VIIRS('VIIRS', now, chFNm_FRP, chFNmOut, chFNmHeight, log)
#                                 spp.log(os.path.join(chDirLog,now.strftime('log_%Y%m%d_%H%M.log'))))
        if not gv_375_r.pick_granule_data_IS4FIRES_v3_0(ifHeightCorrection):
            print('Failed pick the downsampled granule:', chFNmOut)
            return -2
        dLon = gv_375.lon - gv_375_r.lon
        dLon[dLon > 180] -= 360
        dLon[dLon < -180] += 360
        dLon_km = dLon * np.cos(gv_375_r.lat * spp.degrees_2_radians) * 111.
        dLat = gv_375.lat - gv_375_r.lat
        if gv_375_r.height is None: h = -999
        else: h = gv_375_r.height 
        log.log('verification: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
                (gv_375_r.DS_factor_track, gv_375_r.DS_factor_scan, 
                 np.mean(gv_375_r.lon), np.mean(gv_375_r.lat), np.max(h))
                + ' med_Dlon_abs= %gkm, med_Dlat_abs= %gkm, ' % 
                (np.median(np.abs(dLon_km)), np.median(np.abs(dLat) * 111.)) 
                + ' min_Dlon= %gkm, max_Dlon= %gkm, min_Dlat= %gkm, max_Dlat= %gkm,' % 
                (np.min(dLon_km), np.max(dLon_km), 
                 np.min(dLat) * 111., np.max(dLat) * 111.)
                + ' 001p_Dlon= %gkm, 001p_Dlat= %gkm, 9999p_Dlon= %gkm, 9999p_Dlat= %gkm' % 
                (np.percentile(dLon_km,0.01),
                 np.percentile(dLat,0.01) * 111.,
                 np.percentile(dLon_km,99.99),
                 np.percentile(dLat,99.99) * 111.))

        print('Done verification. Time used:', dt.datetime.now() - timer_start_verify)
        ifOK =  (correction_sensitivity_m > np.max(dLon_km) * 1e3 and
                 correction_sensitivity_m > -np.min(dLon_km) * 1e3 and 
                 correction_sensitivity_m > np.max(dLat) * 111000 and
                 correction_sensitivity_m > -np.min(dLat) * 111000)
    else: ifOK = True  # if check is not ordered, just proceed
    
    if ifOK:
        log.log('Successfull downsampling for: ' + gv_375.get_geo_FNm() + '\n')
        spp.ensure_directory_MPI(os.path.join(os.path.split(gv_375.get_geo_FNm())[0], '..', 'downsampled'))
        shutil.move(gv_375.get_geo_FNm(),
                    os.path.join(os.path.split(gv_375.get_geo_FNm())[0], '..','downsampled', 
                                 os.path.split(gv_375.get_geo_FNm())[1]))
        return True
    else:
        log.log('################# >>>>>> Failed downsampling\n')
        return False


############################################################################################

def evaluate_downsampling(chFNm_geo375, chFNm_geo750, chFNm_FRP_, chFNmHeight, tStart, nSteps, 
                          chDirLog, chDirOut, ifDS_Cartesian, ifHeight, ifDraw, mpirank, mpisize):
    #
    # Scans the given time period comparing the downscaling skills for various DS_factor values
    #
    # Histogram of the error
    #
    hist_err_bins = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.02, -0.01,
                              0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    hist_err = np.zeros(shape=(2,hist_err_bins.shape[0]-1))  # lon, lat
    iProcess = 0
    for i in range(nSteps):   # VIIRS has 6 minutes time step

        iProcess += 1
        if np.mod(iProcess-1, mpisize) != mpirank:  continue
        
        now = tStart + spp.one_minute * i * 6
        chFNm_Log = os.path.join(chDirLog, 
                                 now.strftime('log_%Y%j_%Y.%m.%d_%H%M') + '_mpi%02g.txt' % mpirank)
        #
        # Reference granule
        #
        chFNm_FRP = chFNm_FRP_
#        chFNm_FRP = 'd:\\data\\satellites\\VIIRS\\VNP14\\2022.05.01\\VNP14IMG.A2022121.1718.001.2022122012909.nc'
        gv_ref = granule_VIIRS('SNPP', now, chFNm_FRP, chFNm_geo375, chFNmHeight, spp.log(chFNm_Log))
        if not gv_ref.pick_granule_data_IS4FIRES_v3_0(): 
            continue
        if ifDraw:
            spp.ensure_directory_MPI(os.path.join(chDirOut, 'pics_ref',
                                                  gv_ref.now().strftime('VNP03IMGLL\\%Y.%m.%d')))
            gv_ref.draw_granule(os.path.join(chDirOut, 'pics_ref',
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
#                                    os.path.join(chDirOut, 'ref',
#                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
#                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
            t = dt.datetime.now()
#            gv_375.downsample_granule_geo_lonlat(iFactor)
            gv_375.downsample_granule_geo_Cartesian(iFactor_track, iFactor_scan, ifDS_Cartesian)
            gv_375.geolocation_2_nc(True,         # downsampled geodata
                                    os.path.join(chDirOut, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan),
                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
#            gv_375.restore_granule_geo_lonlat()
            true_height = gv_375.restore_granule_geo_Cartesian(ifHeight)
            print('Not storing restored granule')
#            gv_375.geolocation_2_nc(False,
#                                    os.path.join(chDirOut, 'DS_%02i_%02i_restored' % (iFactor_track, iFactor_scan),
#                                                 gv_375.now().strftime('VNP03IMGLL\\%Y.%m.%d'),
#                                                 os.path.split(gv_375.get_geo_FNm())[1] + '.nc4'))
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
            # Get histogram
            #
            h_tmp, bin_edges = np.histogram(dLon_km, hist_err_bins)
            hist_err[0,:] += h_tmp
            h_tmp, bin_edges = np.histogram(dLat*lat2km, hist_err_bins)
            hist_err[1,:] += h_tmp
            #
            # Draw
            #
            if ifDraw:
                spp.ensure_directory_MPI(os.path.join(chDirOut, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_375.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_375.draw_granule(os.path.join(chDirOut, 'DS_%02i_%02i' % (iFactor_track, iFactor_scan), 'pics',
                                             gv_375.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                             os.path.split(gv_375.get_geo_FNm())[1] + '.png'))
                spp.ensure_directory_MPI(os.path.join(chDirOut, 'DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d')))
                gv_375.draw_granule_diff(gv_ref, ['375m','DS_%g_%g' % (iFactor_track, iFactor_scan)],
                                     os.path.join(chDirOut, 'DS_%02i_%02i_diff' % (iFactor_track, iFactor_scan), 'pics',
                                                  gv_ref.now().strftime('VNP03MODLL\\%Y.%m.%d'),
                                                  os.path.split(gv_ref.get_geo_FNm())[1] + '.png'))
    #
    # Draw histogram
    #
    chHeight = {True:'height', False:'noheight'}[ifHeight]
    fig,ax = mpl.pyplot.subplots(1,1)
    ax.plot(0.5*(hist_err_bins[:-1] + hist_err_bins[1:]), hist_err[0], linestyle='-', marker='.',
               linewidth=2.0, label='lon, out of range: %g' % 
               (np.sum(dLon_km >= hist_err_bins[-1]) + np.sum(dLon_km < hist_err_bins[0])))
    ax.plot(0.5*(hist_err_bins[:-1] + hist_err_bins[1:]),hist_err[1], linestyle='-', marker='.',
               linewidth=2.0, label='lat, out of range: %g' % 
               (np.sum(dLat >= hist_err_bins[-1]/lat2km) + np.sum(dLat < hist_err_bins[0]/lat2km)))
    ax.legend()
    ax.set_yscale('log')
    ax.grid('both')
    ax.set_xlabel('km')
    ax.set_ylabel('nbr of pixels')
    ax.set_title('Histogram of error' + {True:', height-corrected', False:' no height correction'}[ifHeight])
    mpl.pyplot.savefig(os.path.join(chDirOut,'histogram_%s.png' % chHeight))
    fig.clf()
    mpl.pyplot.close()
    #
    # Store the histogram
    #
    fOut = open(os.path.join(chDirOut,'histogr_%s.out' % chHeight),'w')
    fOut.write('bins: ' + ' '.join((str(v) for v in hist_err_bins)) + '\n')
    fOut.write('longitude: ' + ' '.join((str(v) for v in hist_err[0,:])) + '\n')
    fOut.write('latitude: ' + ' '.join((str(v) for v in hist_err[1,:])) + '\n')
    fOut.close()


############################################################################
    
def verify_geo_downsample(chFNm_FRP, chFNm_geo375, chFNm_downsampled, now, accuracy, log):
    #
    # Verifies the downsampling for the given time
    # Reads the original and the downsampled granules and compares lon & lat fields
    # The maximum error must be lower than given accuracy. Returns True if it is
    #
    timer_start = dt.datetime.now()
    # Read the downsampled granule
    gv_downsampled = granule_VIIRS('VIIRS', now, chFNm_FRP, chFNm_downsampled, None, log)
    if not gv_downsampled.pick_granule_data_IS4FIRES_v3_0(False): 
        log.log(now.strftime('FAILED reading downsampled granule: %Y%m%d_%H%M, ' + chFNm_downsampled))
        return (STATUS_MISSING_FRP_FILE, None)
    # Read the original granule
    gv_375 = granule_VIIRS('VIIRS', now, chFNm_FRP, chFNm_geo375, None, log)
    if not gv_375.pick_granule_data_IS4FIRES_v3_0(False): 
        log.log(now.strftime('FAILED reading original granule: %Y%m%d_%H%M, ' + chFNm_geo375))
        return (STATUS_MISSING_GEO_FILE, None)
    dLon = gv_375.lon - gv_downsampled.lon
    dLon[dLon > 180] -= 360
    dLon[dLon < -180] += 360
    dLon_km = dLon * np.cos(gv_downsampled.lat * spp.degrees_2_radians) * 111.
    dLat = gv_375.lat - gv_downsampled.lat
    if gv_downsampled.height is None: h = -999
    else: h = gv_downsampled.height
    
    log.log('verification: DS_factor= (%gx%g), lon= %g, lat= %g, heightMAX= %gm,' % 
            (gv_downsampled.DS_factor_track, gv_downsampled.DS_factor_scan, 
             np.mean(gv_downsampled.lon), np.mean(gv_downsampled.lon), np.max(h))
            + ' med_Dlon_abs= %gkm, med_Dlat_abs= %gkm, ' % 
            (np.median(np.abs(dLon_km)), np.median(np.abs(dLat) * 111.)) 
            + ' min_Dlon= %gkm, max_Dlon= %gkm, min_Dlat= %gkm, max_Dlat= %gkm,' % 
            (np.min(dLon_km), np.max(dLon_km), 
             np.min(dLat) * 111., np.max(dLat) * 111.)
            + ' 001p_Dlon= %gkm, 001p_Dlat= %gkm, 9999p_Dlon= %gkm, 9999p_Dlat= %gkm' % 
            (np.percentile(dLon_km,0.01),
             np.percentile(dLat,0.01) * 111.,
             np.percentile(dLon_km,99.99),
             np.percentile(dLat,99.99) * 111.))

    print('Done verification. Time used:', dt.datetime.now() - timer_start)

    if (accuracy > np.max(dLon_km) * 1e3 and accuracy > -np.min(dLon_km) * 1e3 and 
        accuracy > np.max(dLat) * 111000 and accuracy > -np.min(dLat) * 111000):
        log.log('Successfull downsampling for: ' + gv_375.get_geo_FNm() + '\n')
        return (STATUS_SUCCESS, gv_375.get_geo_FNm())
    else:
        log.log('################# >>>>>> Failed downsampling for'  + gv_375.get_geo_FNm())
        return (STATUS_BAD_FILE, gv_375.get_geo_FNm())


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
granVIIRS_ref = granule_VIIRS()
dS = granVIIRS_ref.dS
dT = granVIIRS_ref.dT
theta = granVIIRS_ref.theta


##########################################################################
##########################################################################
##########################################################################
##########################################################################

if __name__ == '__main__':
    #
    # test the VIIRS data
    #
    # Laptop, external disk
    dirMain = 'd:\\data'
    chFNm_FRP = os.path.join(dirMain,'VIIRS_1', 'VNP14','%Y.%m.%d','VNP14IMG.A%Y%j.%H%M.001.*.nc')
    chFNm_geo750 = os.path.join(dirMain, 'VNP03MODLL','VNP03','%Y.%m.%d','VNP03MODLL.A%Y%j.%H%M.001.*.h5')
    chFNm_geo375 = os.path.join(dirMain, 'VIIRS_1','VNP03IMGLL','VNP03','%Y.%m.%d','VNP03IMGLL.A%Y%j.%H%M.001.*.h5')
    chFNm_height = os.path.join(dirMain, 'satellites','VIIRS','height','height_1km.nc4')
    chDirOut = os.path.join(dirMain,'satellites','VIIRS','VNP03IMGLL_compressed')

    ifDraw = False
    if_DS_Cartesian = True
    
    ifGeometry = True
    ifTest_4height = False
    ifHeight_4restore = False
    ifEvaluate_DS = False
    ifGetGlobal_Height = False
    ifLogSummary = False
    ifFitPixel_size = False
    ifDownsample = False
    ifVerify_downscaling = False
    #
    # Draw the basic features of the satellite geometry
    #
    if ifGeometry:
        granule = granule_VIIRS(now_UTC=dt.datetime(2023,3,5,8,55))
        granule.draw_pixel_size(os.path.join(dirMain, 'VIIRS_pixel.png'),
                                ifOverlap_map=True) 
    #
    # Evaluate the skills of the downsampling procedure
    #
    if ifEvaluate_DS:
        spp.ensure_directory_MPI(os.path.join(chDirOut,'log_20230311_height')) 
        spp.ensure_directory_MPI(os.path.join(chDirOut,'log_20230311_noheight')) 
        evaluate_downsampling(chFNm_geo375, chFNm_geo750, chFNm_FRP, chFNm_height,
                              dt.datetime(2022,3,21), 40, 
                              os.path.join(chDirOut,'log_20230311_height'), chDirOut,  
                              if_DS_Cartesian, True, ifDraw)
#        evaluate_downsampling(chFNm_geo375, chFNm_geo750, chFNm_FRP, chFNm_height,
#                              dt.datetime(2022,3,21), 40, 
#                              os.path.join(chDirOut,'log_20230311_noheight'), chDirOut,  
#                              if_DS_Cartesian, False, ifDraw)
    #
    # Get the global IMG height field
    #
    if ifGetGlobal_Height: 
        height,chRes,deg2idx,chOutDir,now = get_IMG_height_calculate(chFNm_geo375, chFNm_FRP, 
                                                                     dt.datetime(2022,5,1), 960,
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

    #
    # Downsample the granule
    # Conclusions from testing:
    # Optimal downsampling is 16 x 16. Smaller scales lead to too large downsampled fields
    # More aggressive scale causes too large and varying correction fields. In both cases the
    # final file size is larger.
    # Height-related correction is NOT a good idea: (i) requires extra field height_DS, reduces power
    # of the corrections just a bit, but noticeably reduces quality of restoration due to incorrect
    # initial positioning of the restored fields. And it is also much slower.
    #
    if ifDownsample:
        
        # 
#        tStart = dt.datetime(2021,4,11)
#        tEnd = dt.datetime(2021,4,24)
#        chFNm_geo375 = os.path.join(dirMain, 'VIIRS_2','VNP03','%Y.%m.%d','VNP03IMGLL.A%Y%j.%H%M.001.*.h5')

        # REDO
        tStart = dt.datetime(2022,3,9,16,42)
        tEnd = dt.datetime(2022,3,9,16,48)
        chFNm_geo375 = os.path.join(dirMain, 'VIIRS_1_1','REDO_downsampling','VNP03IMGLL','VNP03','%Y.%m.%d','VNP03IMGLL.A%Y%j.%H%M.001.*.h5')

        spp.ensure_directory_MPI(os.path.join(chDirOut, 'log_downsample_lonlat_REDO'))
        log_main = spp.log(os.path.join(chDirOut, 'log_downsample_lonlat_REDO',
                                        'log_DS_mpi%i.txt' % mpirank_loc))

        nSteps = round((tEnd - tStart) / dt.timedelta(minutes=6))
        iProcess = 0
        for i in range(nSteps):   # VIIRS has 6 minutes time step
            iProcess += 1
            if np.mod(iProcess-1, mpisize_loc) != mpirank_loc:  continue
        
            now = tStart + spp.one_minute * i * 6
            if not downsample_geo_granule(chFNm_geo375, chFNm_FRP, chFNm_height, now, 
                                          16, 16, 40.0,      # iFactor_track, iFactor_scan, correction_sensitivity_m
                                          chDirOut,            # chDirOut
                                          False, True,      # ifHeightCorrection, ifVerify, 
                                          log_main):
                log_main.log('######### FAILED downsample: ' + chFNm_geo375)
            else:
                print('')
    #
    # A final step before deleting the granule.
    # Runs through the given time period, reads original and compressed granules.
    # In case of successful comparison of the fields, deletes the originals 
    #
    if ifVerify_downscaling:
        ifDelete = False    # if successful evaluation, do we delete the original?
        tStart = dt.datetime(2021,9,16)
        tEnd = dt.datetime(2022,3,7)
        chFNm_geo375 = os.path.join(dirMain, 'VIIRS_3','VNP03IMGLL','VNP03','downsampled','VNP03IMGLL.A%Y%j.%H%M.001.*.h5')
        chDirOut = os.path.join(dirMain,'satellites','VIIRS','VNP03IMGLL_compressed')
        spp.ensure_directory_MPI(os.path.join(chDirOut, 'log_downsample_verify'))
        log_main = spp.log(os.path.join(chDirOut, 'log_downsample_verify',
                                        'log_DS_verify_mpi%i.txt' % mpirank_loc))
        nSteps = round((tEnd - tStart) / dt.timedelta(minutes=6))
        iProcess = 0
        for i in range(nSteps):   # VIIRS has 6 minutes time step
            iProcess += 1
            if np.mod(iProcess-1, mpisize_loc) != mpirank_loc:  continue
        
            redo_list = open(os.path.join(chDirOut, 'log_downsample_verify',
                                          'redo_list_mpi%03i.txt' % mpirank_loc),'w')
            now = tStart + spp.one_minute * i * 6
            iStatus, chFNm_orig = verify_geo_downsample(chFNm_FRP, chFNm_geo375,   # FRP and original
                                                        os.path.join(chDirOut,     # downsampled
                                                                     now.strftime('%Y.%m.%d'),
                                                                     os.path.split(chFNm_geo375)[1] + '.nc4'),
                                                        now, 40.0, log_main)     # accuracy_m
            if iStatus == STATUS_SUCCESS:
                log_main.log(now.strftime('%Y%m%d_%H%M success: ' + chFNm_geo375 + '\n'))
                if ifDelete:
                    os.remove(chFNm_geo375)
            elif iStatus == STATUS_BAD_FILE:   # broken file
                log_main.log(now.strftime('%Y%m%d_%H%M BADBADBAD =======>>>>: ' + chFNm_geo375 + '\n'))
                redo_list.write(now.strftime('BAD FILE %Y%m%d ' + chFNm_geo375 + '\n'))
            elif iStatus == STATUS_MISSING_GEO_FILE:
                log_main.log(now.strftime('%Y%m%d_%H%M MISSING GEO FILE =======>>>>: ' + chFNm_geo375 + '\n'))
                redo_list.write(now.strftime('MISSING GEO FILE %Y%m%d ' + chFNm_geo375 + '\n'))
            elif iStatus == STATUS_MISSING_FRP_FILE:
                log_main.log(now.strftime('%Y%m%d_%H%M MISSING FRP FILE =======>>>>: ' + chFNm_geo375 + '\n'))
                redo_list.write(now.strftime('MISSING FRP FILE %Y%m%d ' + chFNm_geo375 + '\n'))
            else:
                log_main.log(now.strftime('%Y%m%d_%H%M UNKNOWN =======>>>>: ' + chFNm_geo375 + '\n'))
                redo_list.write(now.strftime('UNKNOWN %Y%m%d ' + chFNm_geo375 + '\n'))
        redo_list.close()
