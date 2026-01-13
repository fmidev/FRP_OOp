
'''
This module handles MODIS/VIIRS/SLSTR/... files with FRP and cloud data:
MOD14/MYD14 and MOD35/MYD35, VJ114/VNP14, VJ103/VNP03, ...respectively
The files are supposed to be in netCDF format, whether 3 or 4, obtained 
by applying h4tonccf or h4tonccf_nc4 to the original MODIS hdf files or
from other suitable formats of other satellites. 
The outcome is the consistent set of fire information.

Concept of IS4FIRES v.3.0
Each pixel reported by any satellite can have three status types:
- cloud pixel: no onformation available, np.nan is the value in all data fields
- no-fire pixel: clear sky and zero FRP observed; ambigious case, see below
- fire pixel: clear sky, non-zero FRP found

The problem with no-fire pixel is that it may be not zero but below
the detection limit, which depends on the pixel size. MODIS collection 6
suggests the FRP detection threshold being:
FRP_min_observed = 3.56 MW/km2 * pixel_area + 0.54 MW
Therefore, 1 km2 pixel is good for fires above ~4 MW but 10 km2 pixels miss
all fires lelow ~35 MW.
The problem is that the peak of FRP frequency distribution is 7-15 MW, i.e.
pixels >4 km2 miss the main chunk of inforamtion.
For VIIRS, the situation is better but still the concept stays the same

To handle the ambiguity, each no-fire pixel is attributed with the detection 
threshold and thus declares only absence of fires above that value. The system
then considers several ranges of FRP independently and sets np.nan or zero
depending on the detection limit and the considered FRP range.

This concept is propageted up-scale for coarser grids, where the fraction of
np.nan, zero-FRP and non-zero-FRP pixels is counted for each large-grid cell.
A simple threshold for max fraction of nan values is used to decide whether
the grid cell has a meaningful zero or nan.

This division is further used by the fire forecasting module, which generates
the actual emission fluxes, possibly filling-in the gaps in observations.

Created on 12.8.2020
VIIRS extenstion 30.11.2024

@author: M.Sofiev
'''

import numpy as np
import numpy.f2py
#from scipy import interpolate
from toolbox import supplementary as spp, silamfile, namelist
from toolbox import gridtools, MyTimeVars, stations
from support import netcdftime 
import netCDF4 as nc4
import datetime as dt
import os, sys, glob, copy, time, shutil
from os import path
import matplotlib as mpl
import warnings
#from matplotlib import pyplot as plt
from mpl_toolkits import basemap
import pickle
import land_use as LU_module
import granule_MODIS as sgMOD
import granule_VIIRS as sgVIIRS
import granule_SLSTR as sgSLSTR
import FRP_OOp_pixel
import quality_assurance
import fire_records

ifFortranOK = False
try:
    from src import pixel_projection
    ifFortranOK = True
except:
    try:
        import pixel_projection
        ifFortranOK = True
    except: pass

if not ifFortranOK:
    # Attention: sizes of the arrays must be at the end
    fortran_code_pixel_proj = '''
    
subroutine granule_to_grid(ixMap4p, iyMap4p, gran_area1D, gran_nodata, gran_detect_lim, mapOut, &
                         & nxMap, nyMap, nPixS, nPixT)                 ! sizes must be at the end
  !
  ! Does the bulky work of granule projection to the main grid
  !
  implicit none
  !
  ! Imported parameters
  integer*4, dimension(0:nPixT-1, 0:nPixS-1), intent(in) :: ixMap4p, iyMap4p
  real*4, dimension(0:nPixS-1), intent(in) :: gran_area1D
  real*4, dimension(0:nPixT-1,0:nPixS-1), intent(in) :: gran_nodata, gran_detect_lim
  real*4, dimension(0:2, 0:nyMap-1, 0:nxMap-1), intent(out) ::  mapOut  ! _area, _nodata, _detect_lim
  ! sizes must be at the end
  integer*4, intent(in) :: nxMap, nyMap, nPixS, nPixT
  !
  ! Local variable
  integer :: i, j
  !
  ! nullify output map
  mapOut(:,:,:) = 0.0
  !
  ! fill-in what needs to be: scan the granula nPixS along swath, nPixT along trajectory 
  do i = 0, nPixS-1
    do j = 0, nPixT-1
      if(.not. (iyMap4p(j,i) >= 0 .and. ixMap4p(j,i) >= 0)) cycle
      mapOut(0,iyMap4p(j,i),ixMap4p(j,i)) = mapOut(0,iyMap4p(j,i),ixMap4p(j,i)) + gran_area1D(i)
      mapOut(1,iyMap4p(j,i),ixMap4p(j,i)) = mapOut(1,iyMap4p(j,i),ixMap4p(j,i)) + gran_nodata(j,i)
      mapOut(2,iyMap4p(j,i),ixMap4p(j,i)) = mapOut(2,iyMap4p(j,i),ixMap4p(j,i)) + &
                                          & gran_detect_lim(j,i) * gran_area1D(i)
    end do  ! nPixT along trajectory
  end do  ! nPixS along swath
  !
  ! Normalise the detection limit with the area: it is averaged per-granule
  ! I want empty areas very high
  mapOut(2,:,:) = (mapOut(2,:,:) + 1e-5) / (mapOut(0,:,:) + 1e-15)
end subroutine granule_to_grid


!================================================================================================

subroutine fires_2_unified_map(FP_hour, FP_LU, FP_ix, FP_iy, FP_frp, LU_diurnal, &
                             & detect_lim_lst, obsNightHrs, obsDayHrs, &
                             & fire_list, nNightHrs, nDayHrs, nLUs, nFires) 
  !
  ! Goes through the list of fires and adds them to the unified map
  !
  implicit none
  
  ! Imported parameters
  integer, dimension(0:nFires-1), intent(in) :: FP_hour, FP_ix, FP_iy, FP_LU
  real*4, dimension(0:nFires-1), intent(in) :: FP_frp
  real*4, dimension(0:23, 0:nFires-1), intent(in) :: detect_lim_lst
  real*4, dimension(0:23,0:nLUs-1), intent(in) :: LU_diurnal
  integer, dimension(nNightHrs), intent (in) :: obsNightHrs
  integer, dimension(nDayHrs), intent (in) :: obsDayHrs
  integer, intent(in) :: nNightHrs, nDayHrs, nFires, nLUs
  !
  ! Output is an array 5xnGriddedFires. 5 = ix, iy, LU, FRP, FRP_ratio
  real*4, dimension(0:27,0:nFires-1), intent(out) :: fire_list
  
  ! Local variable
  integer :: iFire, iGRFire, iLU, iHr, nGriddedFires, iCnt_obs, iCnt_noobs, idx
  real*4, dimension(0:23,0:nFires-1) :: fire_tmp, detect_lim_tmp  !, missing_tmp
  real*4, dimension(0:23) :: variationCell
  real*4 :: meanDay, meanNight, R_dn_regul, LU_ratio, fNeg, a, b, V_Sday, V_Snight, &
          & dayMeanNew, DL_day, DL_night, meanDayNew_noobs, meanDayNew_obs, scaleNoObs, scaleObs
  logical :: ifFound
  
  ! Run through the fires and create a list of unique trios: ix, iy, LU
  ! The problem of quadratic scaling but the full map is too large to use 
  !
  nGriddedFires = 0
  fire_tmp = 0.0
  do iFire = 0, nFires-1
    ifFound = .false.
    do iGRFire = 0, nGriddedFires-1
      if(FP_ix(iFire) == nint(fire_list(0,iGRFire)) .and. &
       & FP_iy(iFire) == nint(fire_list(1,iGRFire)) .and. &
       & FP_LU(iFire) == nint(fire_list(2,iGRFire)))then
        fire_tmp(FP_hour(iFire),iGRFire) = fire_tmp(FP_hour(iFire),iGRFire) + FP_frp(iFire)
        ifFound = .true.
!        print *, 'Adding to cell iFire, iGRFire, (ix,iy,iLU, iHr), FRP, FRP_cell:', &
!                & iFire, iGRFire, FP_ix(iFire), FP_iy(iFire), FP_LU(iFire), FP_hour(iFire), FP_frp(iFire), &
!                & fire_tmp(:,iGRFire)
        exit
      endif  ! if got gridded fire
    end do    ! search in gridded fires
    if(.not. ifFound)then
       fire_list(0,nGriddedFires) = FP_ix(iFire)
       fire_list(1,nGriddedFires) = FP_iy(iFire)
       fire_list(2,nGriddedFires) = FP_LU(iFire)
       fire_tmp(FP_hour(iFire),nGriddedFires) = FP_frp(iFire)
       detect_lim_tmp(:,nGriddedFires) = detect_lim_lst(:,iFire)
!       missing_tmp(:,nGriddedFires) = missing_lst(:,iFire)
       nGriddedFires = nGriddedFires + 1                                        
    endif
  end do    ! input fires
  !
  ! Create the list of grid cells with fires.
  ! 24 hours change to 2 params: mean daily and OBSERVED ratio min/max. Where min or max
  ! is zero, will use default for this LU - filled-in later by Python
  !
  ! Now, run over the created list of gridded fires turning the frp_tmp into frp_mean and ratio
  !
  do iGRFire = 0, nGriddedFires-1
    !
    ! Compute mean night-time for observed FRP and default profile 
    !
    iLU = nint(fire_list(2,iGRFire))
    meanNight = 0
    DL_night = 0
    iCnt_obs = 0
    iCnt_noobs = 0
    V_Snight = 0
    do idx = 1, size(obsNightHrs)
      iHr = obsNightHrs(idx)
      V_Snight = V_Snight + LU_diurnal(iHr, iLU)
      if(fire_tmp(iHr,iGRFire) > 0.)then
        meanNight = meanNight + fire_tmp(iHr,iGRFire)
        iCnt_obs = iCnt_obs + 1
      else 
        if(detect_lim_lst(iHr,iGRFire) < 9.99e5) then
          DL_night = DL_night + detect_lim_lst(iHr,iGRFire)
          iCnt_noobs = iCnt_noobs + 1
        endif
      endif   ! FRP > 0
    end do  ! obsNightHrs
    if(iCnt_obs > 0) meanNight = meanNight / iCnt_obs
    if(iCnt_noobs > 0) DL_night = DL_night / iCnt_noobs
    V_Snight = V_Snight / 24.0        !  ! 24. Notebook 11a: sum(LU_diurnal) = 24 we need = 1
    !
    ! Mean values for daytime
    !
    meanDay = 0
    DL_day = 0
    iCnt_obs = 0
    iCnt_noobs = 0
    do idx = 1, size(obsDayHrs) 
      iHr = obsDayHrs(idx)
      if(fire_tmp(iHr,iGRFire) > 0.)then
        meanDay = meanDay + fire_tmp(iHr,iGRFire)
        iCnt_obs = iCnt_obs + 1
      else 
        if(detect_lim_lst(iHr,iGRFire) < 9.99e5) then
          DL_day = DL_day + detect_lim_lst(iHr,iGRFire)
          iCnt_noobs = iCnt_noobs + 1
        endif
      endif   ! FRP > 0
    end do  ! day_hrs
    if(iCnt_obs > 0) meanDay = meanDay / iCnt_obs
    if(iCnt_noobs > 0) DL_day = DL_day / iCnt_noobs
    V_Sday = sum(LU_diurnal(obsDayHrs,iLU)) / 24.0    ! 24. N.11a: sum(LU_diurnal) = 24 we need = 1
    LU_ratio = V_Sday / V_Snight
    !
    ! Compute day-night ratio and regularise with the default one: take mean geometric
    !
    if(meanDay * meanNight > 0.0)then
      !
      ! Both day and night ries are observed. Can compute the observed ratio
      !
      R_dn_regul = sqrt(meanDay / meanNight * LU_ratio)    ! Always regularize
      !
      ! New variation of cell FRP = a * LU_variation + b
      ! a and b are so that R_dn_regul is conserved
      !
      a = (R_dn_regul - 1.) / (V_Sday - 1. - (R_dn_regul * (V_Snight - 1.)))
      if(a < 0) a = 1.0
      b = (1. - a)
      variationCell(:) = a * LU_diurnal(:,iLU) + b
      !
      ! Make sure that the variation is not too low in its minimum
      !
      iCnt_obs = 0
      do while(any(variationCell < 0.0099))   ! 0.01 cannot be used due to numerics
!        print *, variationCell
        fNeg = abs(sum(min(variationCell, 0.01) - 0.01))
        do iHr = 0, 23
          if(variationCell(iHr) < 0.01) then
            variationCell(iHr) = 0.01 - fNeg / 24.
          else
            variationCell(iHr) = variationCell(iHr) - fNeg / 24.
          endif
        end do  ! hr 0..23
        if(iCnt_obs > 10000)then
          if(any(variationCell < 0.005))then
            print *, 'ERRORORORORORORORORO, variationCell', variationCell(:)
            variationCell(:) = LU_diurnal(:,iLU)
          endif
          exit
        endif  ! if too many iterations
        iCnt_obs = iCnt_obs + 1
      end do   ! while variation low point is too low

    else
      !
      ! One of obs times is missing, take default variation
      !
      variationCell(:) = LU_diurnal(:,iLU)

    endif  ! both day and night fires exist
    !
    ! mean daily FRP
    !
    fire_list(3,iGRFire) = 0.
    iCnt_obs = 0
    do iHr = 0, 23
      if(fire_tmp(iHr,iGRFire) > 0.0)then
        fire_list(3,iGRFire) = fire_list(3,iGRFire) + fire_tmp(iHr,iGRFire) / variationCell(iHr)
        iCnt_obs = iCnt_obs + 1
      endif
    end do
    if(iCnt_obs > 0) fire_list(3,iGRFire) = fire_list(3,iGRFire) / iCnt_obs
    !
    ! Make sure that the nighttime observation with missing daytime
    ! does not create extraordinary fire after scaled with default diurnal variation
    ! Unequivocal limit is 100 MW of daypeak, an ~90%-tile of FRP
    !
    if(meanDay == 0.) then   ! meaning, Night > 0 and there is a chance for a huge day spike
      !
      ! Check what happened during day: predicted FRP cannot be much larger than detection limit
      !
      dayMeanNew = fire_list(3,iGRFire) * sum(variationCell(obsDayHrs)) / 3.
      if(dayMeanNew > DL_day) then
        variationCell = (variationCell - 1.) * sqrt(DL_day / dayMeanNew) + 1.
      endif
    else
      !
      ! Both observations are available: the day-night span has been adjusted,
      ! now also account for detection limit
      !
      meanDayNew_obs = 0
      meanDayNew_noobs = 0
      iCnt_obs = 0
      iCnt_noobs = 0
      do idx = 1, size(obsDayHrs) 
        iHr = obsDayHrs(idx)
        if(fire_tmp(iHr,iGRFire) > 0.)then
          meanDayNew_obs = meanDayNew_obs + fire_list(3,iGRFire) * variationCell(iHr)
          iCnt_obs = iCnt_obs + 1
        else 
          if(detect_lim_lst(iHr,iGRFire) < 9.99e5) then
            meanDayNew_noobs = meanDayNew_noobs + fire_list(3,iGRFire) * variationCell(iHr)
            iCnt_noobs = iCnt_noobs + 1
          endif
        endif   ! FRP > 0
      end do  ! day_hrs
      if(iCnt_obs > 0) then
        scaleObs = meanDayNew_obs / (iCnt_obs * meanDay)
      else
        scaleObs = 1.
      endif
      if(iCnt_noobs > 0) then
        scaleNoObs = meanDayNew_noobs / (iCnt_noobs * DL_day)
      else
        scaleNoObs = 1
      endif
      !
      ! Scaling comes from observed and non-observed times
      !
      variationCell = (variationCell - 1.) / sqrt(max(1., sqrt(scaleNoObs * scaleObs))) + 1.

    endif  ! if meanDay == 0
    !
    ! And, again, the mean daily FRP with the updated diurnal profile
    !

    fire_list(3,iGRFire) = 0.
    iCnt_obs = 0
    do iHr = 0, 23
      if(fire_tmp(iHr,iGRFire) > 0.0)then
        fire_list(3,iGRFire) = fire_list(3,iGRFire) + fire_tmp(iHr,iGRFire) / variationCell(iHr)
        iCnt_obs = iCnt_obs + 1
      endif
    end do
    if(iCnt_obs > 0) fire_list(3,iGRFire) = fire_list(3,iGRFire) / iCnt_obs

!    print *, variationCell
    !
    ! Return the suggested diurnal variation too
    !
    fire_list(4:27,iGRFire) = variationCell(0:23)

!      dayScaledFire = meanNight * LU_diurnal(12,iLU) / LU_diurnal(2,iLU) 
!        dayDetectLim = detect_lim_lst(12,iFire)
!        if(dayScaledFire > dayDetectLim) then
!          ! Reduce the diurnal span: night fire is disproportionally large
!          ! Still, don't overdo: nighttime fire is hardly smaller during the daytime
!          ! Rains etc will be handled separately
!          print *, 'WARNING: high night FRP sum, hrs 0-3, and diurnal ratio:', meanNight, LU_ratio
!          print *, 'dayScaledFire > dayDetectLim:', dayScaledFire, dayDetectLim
!          if(LU_ratio * dayDetectLim / dayScaledFire < 1)then
!            print *, 'Too much: reduce to 1. In theiry, LU_ratio', LU_ratio
!            fire_list(4,iGRFire) = -1.
!            variationCell(:) = 1.
!          else
!            fire_list(4,iGRFire) = -LU_ratio * dayDetectLim / dayScaledFire
!            variationCell(:) = (LU_diurnal(:,iLU) - 1.) * dayDetectLim / dayScaledFire + 1.
!            print *, 'initial variation in cell', LU_diurnal(:,iLU)
!            print *, 'reduced variation in cell:', variationCell(:) 
!          endif  ! relly high nighttime: unity scaling forced
!        endif  ! Too high daytime FRP due to scaling
!      endif  ! meanDay == 0

  end do  ! iGRFire

!  print *, 'Gridded fires'
!  do iGRFire = 0, nGriddedFires-1
!    print *, iGRFire, fire_list(0,iGRFire), fire_list(1,iGRFire), fire_list(2,iGRFire), &
!           & fire_list(3,iGRFire), fire_list(4,iGRFire), fire_tmp(:,iGRFire), &
!           & detect_lim_tmp(:,iGRFire), missing_tmp(:,iGRFire)  
!  end do
!  print *,''

end subroutine fires_2_unified_map


!==================================================================================

subroutine collect_frp_cells(iLU, FPgrid_LU, FPgrid_ix, FPgrid_iy, FPgrid_frp, & 
                           & TSM_cells_idx, detect_lim, tsmVals, nCells, nFires, nx, ny)
  !
  ! Collects the one time step of tsMatrix from the vectors of FRP, its LU and grid indices
  !
  implicit none
  
  ! Imported parameters
  integer, intent(in) :: iLU, nFires, nCells, nx, ny
  integer, dimension(0:nFires-1), intent(in) :: FPgrid_LU, FPgrid_ix, FPgrid_iy
  real*4, dimension(0:nFires-1), intent(in) :: FPgrid_frp
  integer*4, dimension(0:ny-1, 0:nx-1), intent(in) :: TSM_cells_idx
  real*4, dimension(0:ny-1, 0:nx-1), intent(in) :: detect_lim 
  
  ! Output variable
  real*4, dimension(0:nCells-1), intent(out) ::  tsmVals

  ! Local variables
  integer :: iFire, ix, iy

  ! Store the fires into the tsMatrix
  !
  do iFire = 0, nFires-1
    if (iLU < 0 .or. FPgrid_LU(iFire) == iLU)then
      if (TSM_cells_idx(FPgrid_iy(iFire), FPgrid_ix(iFire)) >= 0 .and. &
        & TSM_cells_idx(FPgrid_iy(iFire), FPgrid_ix(iFire)) < nCells)then
        tsmVals(TSM_cells_idx(FPgrid_iy(iFire), FPgrid_ix(iFire))) = &
               & tsmVals(TSM_cells_idx(FPgrid_iy(iFire), FPgrid_ix(iFire))) + FPgrid_frp(iFire)
      else
        print *, 'List index out of list (iFire, ix, iy, iLU, index, nCells):', &
                     & iFire, FPgrid_ix(iFire), FPgrid_iy(iFire), FPgrid_LU(iFire), &
                     & TSM_cells_idx(FPgrid_iy(iFire), FPgrid_ix(iFire)), nCells
      endif
    endif  ! LU is correct
  end do   ! iFire
  !
  ! Fill-in the remaining parts of the tsMatrix with minus-detection limit
  !
  do ix = 0, nx-1
    do iy = 0, ny-1
      if(TSM_cells_idx(iy, ix) >= 0 .and. TSM_cells_idx(iy, ix) < nCells) then
        if(tsmVals(TSM_cells_idx(iy,ix)) == 0) tsmVals(TSM_cells_idx(iy,ix)) = -detect_lim(iy,ix)
      endif 
    end do
  end do
  
end subroutine collect_frp_cells


!==================================================================================

subroutine select_burning_regions(map_frp, map_to_select, nBurn_pixels, fMissing, &
                                & map_selected, nx, ny)
  !
  ! Selects the burning regions setting the rest to missing value
  ! For each pixel, looks for active fires in map_frp in the vicinity of the pixel, as 
  ! defined by nBurn_pixels. If there are fires, the value is copied, if not - set 
  ! to missing value
  !
  implicit none 
  
  ! Imported parameters
  integer, intent(in) :: nBurn_pixels, nx, ny
  real*4, dimension(0:ny-1, 0:nx-1), intent(in) :: map_frp, map_to_select
  real, intent(in) :: fMissing
  
  ! Output map
  real*4, dimension(0:ny-1, 0:nx-1), intent(out) :: map_selected
  
  ! Local variables
  integer :: ix, iy, ixPixMin, ixPixMax, iyPixMin, iyPixMax 
  
  do ix = 0, nx-1
    do iy = 1, ny-1
      ixPixMin = max(0, ix-nBurn_pixels)
      iyPixMin = max(0, iy-nBurn_pixels)
      ixPixMax = min(nx-1, ix+nBurn_pixels)
      iyPixMax = min(ny-1, iy+nBurn_pixels)
      if(sum(map_frp(iyPixMin:iyPixMax, ixPixMin:ixPixMax)) > 0)then
        map_selected(iy, ix) = map_to_select(iy,ix)
      else
        map_selected(iy, ix) = fMissing
      endif  ! is a fire region
    end do   ! iy
  end do  ! ix
end subroutine select_burning_regions


!==================================================================================

subroutine get_prime_contributor(arIX, arIY, arFRP, binsFRP, arContrib, nxGrd, nyGrd, nFires, nFRPbins)
  !
  ! For each grid cell of the map calculates the fraction of the largest fire and build a 
  ! histogram of these contributions as a function of the grid cell FRP
  !
  implicit none
  
  ! Imported parameters
  real, dimension(0:nFires-1), intent(in) :: arFRP
  integer, dimension(0:nFires-1), intent(in) :: arIX, arIY
  integer, intent(in) :: nFires, nFRPbins, nxGrd, nyGrd
  real, dimension(0:nFRPbins-1), intent(in) :: binsFRP 
  real, dimension(0:3, 0:nFRPbins-1), intent(out) :: arContrib ! 0 -> mean fraction
                                                               ! 1 -> mean squared fraction
                                                               ! 2 -> nbr of grid cells included
                                                               ! 3 -> nbr of fires in the cells
  ! Local parameters
  ! mapTmp: 0 -> total FRP in the cell,
  !         1 -> the largest fire FRP
  !         2 -> total number of fires in a cell
  real, dimension(0:2, 0:nxGrd-1, 0:nyGrd-1) :: mapTmp
  integer :: ix, iy, iFire, iBin

  mapTmp = 0.0
  arContrib = 0.0

  ! go through the fires and collect the total sum for each grid cell and the largest element
  !
  do iFire = 0, nFires-1
    mapTmp(0, arIX(iFire), arIY(iFire)) = mapTmp(0, arIX(iFire), arIY(iFire)) + arFRP(iFire)
    mapTmp(2, arIX(iFire), arIY(iFire)) = mapTmp(2, arIX(iFire), arIY(iFire)) + 1
    if(mapTmp(1, arIX(iFire), arIY(iFire)) < arFRP(iFire)) then
      mapTmp(1, arIX(iFire), arIY(iFire)) = arFRP(iFire)
    endif
  end do   ! iFire
  !
  ! Having the map populated, calculate the add-ons to the output arrays:
  ! - fraction of the largest fire in the total in all non-zero grid cells
  ! - square of this fraction - for the future standard deviation
  !
  do iy = 0, nyGrd-1
    do ix = 0, nxGrd-1
      if(mapTmp(0,ix,iy) > 0)then
        ! FRP is non-zero, find the bin and get the fraction
        iBin = 0
        do while(mapTmp(0,ix,iy) > binsFRP(iBin+1) .and. iBin < nFRPbins-1)
          iBin = iBin + 1
        end do
        ! sum up the count, sum and the squares
        arContrib(0,iBin) = arContrib(0,iBin) + mapTmp(1,ix,iy) / mapTmp(0,ix,iy)
        arContrib(1,iBin) = arContrib(1,iBin) + (mapTmp(1,ix,iy) / mapTmp(0,ix,iy))**2
        arContrib(2,iBin) = arContrib(2,iBin) + 1
        arContrib(3,iBin) = arContrib(3,iBin) + mapTmp(2,ix,iy)
      endif   ! non-zero FRP
    end do  ! nx
  end do  ! iy

end subroutine get_prime_contributor

'''

#    from numpy.distutils.fcompiler import new_fcompiler
#    compiler = new_fcompiler(compiler='intel')
#    compiler.dump_properties()

    # Compile the library and, if needed, copy it from the subdir where the compiler puts it
    # to the current directory
    #
    print(np.f2py.rules)
    vCompiler = np.f2py.compile(fortran_code_pixel_proj, modulename='pixel_projection', 
                                verbose=1, extension='.f90')
    if vCompiler == 0:
        cwd = os.getcwd()     # current working directory
        if path.exists(path.join('pixel_projection','.libs')):
            list_of_files = glob.glob(path.join('pixel_projection','.libs','*'))
            latest_file = max(list_of_files, key=path.getctime)
            shutil.copyfile(latest_file, path.join(cwd, path.split(latest_file)[1]))
        try: 
            from src import pixel_projection
            ifFortranOK = True
        except:
            try:
                import pixel_projection
                ifFortranOK = True
            except:
                ifFortranOK = False

    if not ifFortranOK:
        print('>>>>>>> FORTRAN failed, have to use Python. It will be SLO-O-O-O-O-O-OW')

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
    print('IS4FIRES_v3_0_driver: SLURM pseudo-MPI activated', chMPI, 'tasks: ', mpisize_loc)
except:
    # not in pihti - try usual way
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpisize_loc = comm.size
        mpirank_loc = comm.Get_rank()
        chMPI = '_mpi%03g' % mpirank_loc
        print ('IS4FIRES_v3_0_driver: MPI operation, mpisize=', mpisize_loc, chMPI)
    except:
        print ("IS4FIRES_v3_0_driver: mpi4py failed, single-process operation")
        mpisize_loc = 1
        mpirank_loc = 0
        chMPI = ''
        comm = None


#################################################################################
#
# A debugger switch
#
ifDebug = False


#################################################################################
#
# The timezone class is needed to hold the timezone map to provide quick access to it
# All we need is just to get the local time from UTC
#
#################################################################################

class timezone():
    
    #==================================================================
    
    def __init__(self, tzMap, tzMetaFile, log):
        self.lonRef = 0.0; self.latRef = 0.0
        self.dlon = 1.0;   self.dlat = 1.0
        self.nx = 1;       self.ny = 1
        self.tz_map = np.ones(shape=(self.nx, self.ny))
        self.log = log
        self.log.log('\n\n>>>>>>>>>Timezone class is not yet implemented\n\n')

    #==================================================================

    def local_time(self, now_UTC, lons, lats):
        ix_loc = np.int32(np.round((lons - self.lonRef) / self.dlon))
        iy_loc = np.int32(np.round((lats - self.latRef) / self.dlat))
        ii_loc = iy_loc * self.nx + ix_loc
        self.log.log('\n\n>>>>>>>>>Timezone class is not yet implemented\n\n')
        try:
            return np.array(list( (now_UTC + (spp.one_hour * self.tzMap[ii_loc])))) 
        except:
            self.log.log('Wrong geographical coordinates given:')
            self.log.log('Lon: ' + ' '.join(lons))
            self.log.log('ix: ' + ' '.join(ix_loc))
            self.log.log('Lat: ' + ' '.join(lats))
            self.log.log('iy: ' + ' '.join(iy_loc))
            return [np.nan]


#################################################################################
#
# We need to identify the satellites from their short names and vice versa
#
#################################################################################
    
def satellite_from_shortName(self, shortName):
    #
    # Satellite products that can be handled
    #
    try:
        return {'VNP03IMGLL':('VIIRS','auxiliary'), 'VJ103IMG':('VIIRS','auxiliary'),
                'VNP03IMG':('VIIRS','auxiliary'),   'VNP14IMGLL':('VIIRS','fire'),
                'VJ114IMG':('VIIRS','fire'),        'VNP14IMG':('VIIRS','fire'),
                'MOD03':('MODIS','auxiliary'),      'MYD03':('MODIS','auxiliary'),
                'MOD14':('MODIS','fire'),           'MYD14':('MODIS','fire'),
                }[shortName]
    except:
        raise ValueError('Unknown product: ' + shortName)





#############################################################################
#############################################################################
#
# Class daily_maps fills in and analyses the map of FRP using the information of
# granules from MxD14 and MxD35, i.e. fires and clouds.
# Can use other sources of data too but for now it is for MODIS only
# The main output is a daily map of recorded fires, recorded zeroes and missing grid cells
#
#############################################################################



class daily_maps():

    #========================================================================

    def __init__(self, grid_def=(None,None), day=None, cloud_threshold=None, ifLocalTime=None, 
                 nTimes=24, LU=None, quality_assurance=None, log=None):
        #
        # Constructor can serve also as a formal initializaer for a void instance
        # That gives a chance to read it from netCDF4 file
        # Exception of the log file, which must be actual for this run
        #
        self.grid = copy.deepcopy(grid_def[0])      # main grid of the analysis
        self.gridName = grid_def[1]                 # name of the grid (no spaces)
        self.nTimes = nTimes                        # 24 or 1 for hourly and daily resolution
        if day is None: self.day = None
        else: self.day = dt.datetime(day.year, day.month, day.day,0,0)
        self.cloud_threshold = cloud_threshold      # [0..1] Above that, no data
        self.ifLocalTime = ifLocalTime              # can be UTC or solar local time
        self.LU = LU                                # pre-intialized land_use instance
        self.log = log                           # log file
        self.FP = fire_records.fire_records(self.log)
        self.QA = quality_assurance              # frequent fires, huge fires, nad days, etc
        # if grid is known, initialize the maps. 
        if self.grid is not None: 
            self.init_data_structures(0)  # nFires
        # ad-hoc scale of detection limit with cloudiness: cld=0.3 => DL*1.5; cld=0.6 => DL*10
        self.cld_sigmoid_scale = 75.
        self.cld_sigmoid_slope = 10.
        self.cld_sigmoid_shift = 0.8
#        self.obsNightHrs = np.array([0, 1, 2, 3])  # MODIS overpass, night, LST
        self.obsNightHrs = np.array([0, 1, 2, 3, 22])  # MODIS overpasses, night, LST: 22LST included 
        self.obsDayHrs = np.array([11,12,13])          # MODIS overpass, day, LST


    #=======================================================================

    def from_file(self, chNC4FNm, land_use=None, sensor=None):
        #
        # Initialises the daily map from a daily netCDF file
        # Land use info is _NOT_ stored there. If it will be needed later,
        # someone has to initialize it
        #
        fIn = nc4.Dataset(chNC4FNm,'r')
        fIn.set_auto_mask(False)            ## Never mask
        self.grid, self.gridName = silamfile.read_grid_from_nc(fIn)
        self.cloud_threshold = fIn.cloud_threshold 
        self.ifLocalTime = {'True':True,'False':False}[fIn.getncattr('ifLocalTime')]
        self.day = netcdftime.num2date(fIn.variables['time'][:], fIn.variables['time'].units)[0]
        #
        # Daily files have different set of maps
        #
        ifDailyFile = 'detection_limit_cld_daily' in fIn.variables
        if ifDailyFile:
            try: self.detectLim_clds_daily = fIn.variables['detection_limit_cld_daily'][0,:,:].data.astype(np.float32)
            except: self.detectLim_clds_daily = fIn.variables['detection_limit_cld_daily'][0,:,:].astype(np.float32)
            try: self.detectLim_clrSky_daily = fIn.variables['detection_limit_clearsky_daily'][0,:,:].data.astype(np.float32)
            except: self.detectLim_clrSky_daily = fIn.variables['detection_limit_clearsky_daily'][0,:,:].astype(np.float32)
            try: self.NoData_daily = fIn.variables['missing_cells_daily'][0,:,:].data.astype(np.float32)
            except: self.NoData_daily = fIn.variables['missing_cells_daily'][0,:,:].astype(np.float32)
            self.nTimes = 1
            self.nGriddedFires = fIn.nGriddedFires
            # ...and reserve space for daily unified map, if needed
            try:
                self.unified_map.shape
            except:
                self.unified_map = np.zeros(shape=self.detectLim_clds_daily.shape)
            self.unified_map_base = None
        else:
            try: self.mapArea = fIn.variables['Pixels_tot_area'][:,:,:].data.astype(np.float32)
            except: self.mapArea = fIn.variables['Pixels_tot_area'][:,:,:].astype(np.float32)
            try: self.mapDetectLim = fIn.variables['detection_limit'][:,:,:].data.astype(np.float32)
            except: self.mapDetectLim = fIn.variables['detection_limit'][:,:,:].astype(np.float32)
            try: self.mapNoData = fIn.variables['missing_cells'][:,:,:].data.astype(np.float32)
            except: self.mapNoData = fIn.variables['missing_cells'][:,:,:].astype(np.float32)
            self.nTimes = self.mapDetectLim.shape[0]
        #
        # Read the fire records
        # Careful: fire records will _ADD_ the records from the file to the ones existing, if any.
        # Need to reinitialize
        #
        self.FP = fire_records.fire_records(self.log)
        self.FP.from_nc(fIn, sensor)
        #
        # Daily file has also daily fire lists, gridded
        #
        if ifDailyFile and self.FP.nFires > 0:
            self.FPgrid_frp = fIn.variables['FPgrid_frp'][:].astype(np.float32)
#            self.FPgrid_ratio = fIn.variables['FPgrid_frp_ratio'][:].astype(np.float32)
            self.FPgrid_diurnal = fIn.variables['FPgrid_diurnalvar'][:].astype(np.float32)
            self.FPgrid_LU = fIn.variables['FPgrid_LU'][:].astype(np.int16)
            self.FPgrid_ix = fIn.variables['FPgrid_ix'][:].astype(np.int16)
            self.FPgrid_iy = fIn.variables['FPgrid_iy'][:].astype(np.int16)

        # land use needs to be reset 
        if self.LU is None:
            if land_use is None:
                self.LU = LU_module.land_use(nc_file = chNC4FNm)      # empty class, then read from file
            else:
                self.LU = land_use
        else:
            self.LU.update_from_file(nc_file = chNC4FNm)
        #
        # Are the maps processed by QA routines? Note that old generator does not do that and leaves no traces
        #
        try: 
            self.QA_flag = fIn.QA_flag
        except:
            self.QA_flag = np.int64(0)
        
        self.QA = quality_assurance.QA.from_nc(fIn, self.log)
        
        return self

    #========================================================================
    
    def init_data_structures(self, nFires):
        #
        # Initialises the data structures and reads the land use map, which dimension
        # is needed to get correct size of the FRP map
        #
        self.mapNoData = np.zeros(shape=(self.nTimes, self.grid.ny, self.grid.nx), 
                                  dtype=np.float32)
        self.mapDetectLim = np.ones(shape=(self.nTimes, self.grid.ny, self.grid.nx), 
                                    dtype=np.float32) * 1e10
        self.mapArea = np.zeros(shape=(self.nTimes, self.grid.ny, self.grid.nx), 
                                dtype=np.float32)
#        self.unified_map = np.zeros(shape=(self.nTimes, self.grid.ny, self.grid.nx))

        self.FP.init_data_structures(nFires, self.grid, self.gridName)   # fire records

    #========================================================================
    
    def get_fire_records(self):
        self.FP.QA = self.QA    # pass the QA information
        self.FP.LU_metadata = self.LU.metadataFNm   # pass the land-use information
        self.FP.log = self.log  # pass the log
        return self.FP
        

    #========================================================================
    
#    def Fill_daily_maps(self, satFire_templ, satAux_templ, sources2use, tStep, ifDrawGranules):
    def Fill_daily_maps(self, satFire_files, satAux_files, ifDrawGranules):
        # 
        # Collects the maps for the specific day and stores them into the output
        # Requires an exact list of granules, synchronized fire and auxiliary.
        # Holes and inconsistencies should be handled above.
        #
        # maps with no-fires observations have a problem: different
        # recordings mean different detection limit. Have to take average or 
        # min or max. Needs to be checked.
        # Note the (lat,lon) order: longitude changes fastest
        #
        for fnmFire, fnmAux in zip(satFire_files, satAux_files):

            if ifDebug: self.log.log(now.strftime('%Y%m%d-%H:%M, ' + self.gridName))
            #
            # create the granule accounting for the satellite type
            #
            if (sgMOD.productType(os.path.split(fnmFire)[1])[1] == 'fire' and 
                sgMOD.productType(os.path.split(fnmAux)[1])[1] == 'auxiliary'):
                gran = sgMOD.granule_MODIS(now_UTC = None, 
                                           chFRPfilesTempl = fnmFire, 
                                           chAuxilFilesTempl = fnmAux,
                                           log = self.log)
            elif sgVIIRS.productType(fnmFire)[1] == 'fire' and sgVIIRS.productType(fnmAux)[1] == 'auxiliary':
                gran = sgVIIRS.granule_VIIRS(now_UTC = None, 
                                             chFRPfilesTempl = fnmFire, 
                                             chAuxilFilesTempl = fnmAux,
                                             log = self.log)
            else:
                raise ValueError('fill_daily_maps could not identify the satellite: %s, %s' % (fnmFire, fnmAux))
            
            #
            # get the data
            #
            if not gran.pick_granule_data_IS4FIRES_v3_0():
                self.log.log('Unreadable granule: %s, %s' % (fnmFire, fnmAux))
#                             (path.split(now.strftime(satFire_templ).replace('$SAT',satellite))[1],
#                              path.split(now.strftime(satAux_templ).replace('$SAT',satellite))[1]))
                continue
            self.ifLocalTime = False
            #
            # Draw?
            #
            if ifDrawGranules:
                gran.draw_granule(path.join(path.split(fnmFire)[0], #now.strftime(satFire_templ).
#                                                       replace('$SAT',satellite))[0],
                                                       '..', 
                                                       satellite + 
                                                       gran.now_UTC.strftime('_%Y%m%d_%H%M.png')))
            #
            # Observation operator for this satellite
            #
            OOp = FRP_OOp_pixel.FRP_observation_operator_pixel(sgMOD.productType(os.path.split(fnmAux)[1])[0], self.log)
            #
            # project the granule to the grid, note that FRP is in a separate array
            # Watchout dimensions: 
            # - grid area and detection limit are 1D, granule can have nans.
            # Use broadcasting and subsetting to align all these and prevent
            # nans from penetrating into computations.
            #
            # Put this granule to map
            # fxSwath, fySwath are 1D but reshape is not allowed since idxFinite
            # can change their size
            #
            fxSwath, fySwath = self.grid.geo_to_grid(gran.lon,    #gran.lon_1km,  #[idxFinite], 
                                                     gran.lat)    #gran.lat_1km)  #[idxFinite])
            ixSw = np.round(fxSwath).astype(int)
            # out of grid check and cut
            idxX_OK = np.logical_and(ixSw >= 0, ixSw < self.grid.nx)
            if not np.any(idxX_OK) : continue  # nothing inside the lon-range
            iySw = np.round(fySwath).astype(int)
            idx_OK = np.logical_and(idxX_OK, np.logical_and(iySw >= 0, iySw < self.grid.ny))
            if not np.any(idx_OK) : continue   # nothing inside the grid
            #
            # Add total area, detecion limit etc for each particular grid cell
            # If possible, do it via fortran call: faster
            #
            if ifFortranOK:
#                    timer.start_timer('cycle_f')
                #
                # Count the area of clouds and sunglint.
                # For now, just count the _observed_ area
                #
                # these are the parameters that need to be projected to the grid
#                    gran_area = ((gran.area[None,:] + 
#                              np.zeros(shape=gran.lon_1km.shape,dtype=np.float32)))[idx_OK]
##                              np.zeros(shape=gran.lon_1km.shape,dtype=np.float32))[idxFinite])[idx_OK]
#                    gran_detect_lim = ((gran.area[None,:] *   # broadcasst 1D->2D
#                                    gran.detection_limit()))[idx_OK]
##                                    gran.detection_limit())[idxFinite])[idx_OK]
                gran_nodata = (gran.area[None,:] *   # broadcasst 1D->2D
                               gran.BitFields.QA / 11. *  # cloud: 0, 1
                               gran.BitFields.sunglint)
#                                gran.BitFields.sunglint)[idxFinite])[idx_OK]
                # subset the granule to inside-grid part
                ixSw[np.logical_not(idx_OK)] = -1
                iySw[np.logical_not(idx_OK)] = -1
                #
                # The FORTRAN call
                #
                try:
                    mapTmp = pixel_projection.granule_to_grid(ixSw, iySw, gran.area, gran_nodata, 
                                                              OOp.detection_limit_granule(gran), 
                                                              self.grid.nx, self.grid.ny, 
                                                              ixSw.shape[1], ixSw.shape[0])
                except:
                    self.log.log('Failed FORTRAN granule projection to grid: %s, %s' % (fnmFire, fnmAux))
                    continue
                                                        
                # Area and no-data are summed up 
                self.mapArea[gran.now_UTC.hour,:,:] += mapTmp[0,:,:] 
                self.mapNoData[gran.now_UTC.hour,:,:] += mapTmp[1,:,:]
                # detection limit is min of what exists and what is in the granule 
                self.mapDetectLim[gran.now_UTC.hour,:,:] = np.minimum(self.mapDetectLim[gran.now_UTC.hour,:,:],
                                                                      mapTmp[2,:,:]) 
#                    timer.stop_timer('cycle_f')
#                    timer.report_timers(chTimerName='cycle_f')
            else:
#                    timer.start_timer('cycle_py')
                #
#                    # the idxFinite is 2D: remove nans
#                    idxFinite = np.logical_and(np.isfinite(gran.lon_1km), np.isfinite(gran.lat_1km))
                # subset the granule to inside-grid part
                ixSwOK = ixSw[idx_OK]
                iySwOK = iySw[idx_OK]
                #
                # Count the area of clouds and sunglint.
                # For now, just count the _observed_ area
                #
                # these are the parameters that need to be projected to the grid
                # there used to be area_corr - area of the pixel corrected with the bow-tie overlap
                # but this was probably wrong: no matter how pixels overlap, the sensitivity depends
                # on the true area of each of them
                #
                gran_area = ((gran.area[None,:] + 
                              np.zeros(shape=gran.lon.shape,dtype=np.float32)))[idx_OK]
#                              np.zeros(shape=gran.lon_1km.shape,dtype=np.float32))[idxFinite])[idx_OK]
                gran_detect_lim = ((gran.area[None,:] *   # broadcasst 1D->2D
                                    OOp.detection_limit(gran)))[idx_OK]
#                                    gran.detection_limit())[idxFinite])[idx_OK]
                gran_nodata = ((gran.area[None,:] *   # broadcasst 1D->2D
                                gran.BitFields.QA / 11. *  # cloud: 0, 1
                                gran.BitFields.sunglint))[idx_OK]
#                                gran.BitFields.sunglint)[idxFinite])[idx_OK]
                # a set of unique tuplas wiht the ix and iy coordinates inthe given grid
                gran_grid_cells = np.array(list(set(zip(ixSwOK,iySwOK))))
                for ix, iy in gran_grid_cells:
                    idxThisCell = np.logical_and(ixSwOK == ix, iySwOK == iy)
                    # area of the cell
                    cellArea = np.sum(gran_area[idxThisCell])
                    self.mapArea[gran.now_UTC.hour,iy,ix] += cellArea
                    # clouds and sunglint
                    self.mapNoData[gran.now_UTC.hour,iy,ix] += np.sum(gran_nodata[idxThisCell])
                    # theoretical fire detection limit for the cell is a weighted
                    # sum of detection limits of the pixels
                    # on the cell-side, the new contribution has to be min of the possibilities.
                    self.mapDetectLim[gran.now_UTC.hour,iy,ix] = min(self.mapDetectLim[gran.now_UTC.hour,iy,ix], 
                                                            np.sum(gran_detect_lim[idxThisCell]) / 
                                                            cellArea)
#                    timer.stop_timer('cycle_py')
#                    timer.report_timers(chTimerName='cycle_py')

            # Finally, the fires
            # The are arranged not along the map but as a bunch of 1D vectors
            #
            if gran.nFires > 0:
                self.FP.timezone = 'UTC'    # raw satellite data are in UTC
                self.FP.QA_flag = np.int64(0)  # raw data means no QA
                self.FP.grid = self.grid
                self.FP.timeStart = self.day
                fxFRP, fyFRP = self.grid.geo_to_grid(gran.FP_lon, gran.FP_lat)
                ixFRP = np.round(fxFRP).astype(int)
                iyFRP = np.round(fyFRP).astype(int)
                idxFOK = np.logical_and(np.logical_and(np.logical_and(np.logical_and
                                                                      (gran.FP_frp > 0.0, 
                                                                       ixFRP >= 0), 
                                                                      iyFRP >= 0), 
                                                       ixFRP < self.grid.nx), 
                                        iyFRP < self.grid.ny)
                n = np.sum(idxFOK) 
                if n > 0:
                    self.FP.nFires += n 
                    self.FP.FRP = np.append(self.FP.FRP, gran.FP_frp[idxFOK])
                    self.FP.lon = np.append(self.FP.lon, gran.FP_lon[idxFOK])
                    self.FP.lat = np.append(self.FP.lat, gran.FP_lat[idxFOK])
                    self.FP.dS = np.append(self.FP.dS, gran.FP_dS[idxFOK])
                    self.FP.dT = np.append(self.FP.dT, gran.FP_dT[idxFOK])
                    self.FP.T4 = np.append(self.FP.T4, gran.FP_T4[idxFOK])
                    self.FP.T4b = np.append(self.FP.T4b, gran.FP_T4b[idxFOK])
                    self.FP.T11 = np.append(self.FP.T11, gran.FP_T11[idxFOK])  # channel I5 is 11 um
                    self.FP.T11b = np.append(self.FP.T11b, gran.FP_T11b[idxFOK])
                    self.FP.TA = np.append(self.FP.TA, gran.FP_TA[idxFOK])
                    self.FP.ix = np.append(self.FP.ix, ixFRP[idxFOK])
                    self.FP.iy = np.append(self.FP.iy, iyFRP[idxFOK])
                    self.FP.time = np.append(self.FP.time, np.ones(shape=(n),dtype=np.int64) * (gran.now_UTC - self.day).total_seconds()) # midpoint of the granule
                    self.FP.line = np.append(self.FP.line, gran.FP_line[idxFOK])
                    self.FP.sample = np.append(self.FP.sample, gran.FP_sample[idxFOK])
                    self.FP.SolZenAng = np.append(self.FP.SolZenAng, gran.FP_SolZenAng[idxFOK])
                    self.FP.ViewZenAng = np.append(self.FP.ViewZenAng, gran.FP_ViewZenAng[idxFOK])
                    self.FP.satellite = np.append(self.FP.satellite, gran.FP_satellite[idxFOK])
                    if self.LU is None:
                        self.FP.LU = np.append(self.FP.LU, 'missing')
                    else:
                        self.FP.LU = np.append(self.FP.LU, self.LU.get_LU_4_fires(gran.FP_lon[idxFOK],
                                                                                  gran.FP_lat[idxFOK]))
        #
        # Massage the maps and create a unified map
        #
        # Normalise the maps
        self.mapNoData[self.mapArea > 0] /= self.mapArea[self.mapArea > 0]
        self.mapNoData = 1. - self.mapNoData    # now it is really the missing data
        self.mapDetectLim[self.mapArea == 0] = 0.0

#        #
#        # Unify the maps into the single map:
#        # - non-zero FRP cells retain FRP
#        # - not covered by overpasses cells get NaN
#        # - cloudy cells above threshold get Nan
#        # - zero-recorded pixels get minus detection limit
#        #
#        # Non-observed cells and time moments: no data
#        self.unified_map[:,:,:] = np.where(self.mapArea > 0, 1.0, np.nan)
#        #
#        # clouds etc: no data
#        self.unified_map[self.mapNoData > self.cloud_threshold] = np.nan
#        #
#        # reported zeroes: put the minus-detection-limit
#        self.unified_map *= - self.mapDetectLim
#        #
#        # Finally, FRP, wherever was observed
#        #
#        self.unified_map[self.FP_hour, self.FP_iy, self.FP_ix] += self.FP_frp
        
        
        self.log.log('Stats %s %s: total FRP: %g MW, fire %g, cloud %g, below-detection %g, missing %g out of %g cells' %
                     (self.gridName, self.day.strftime('%Y-%m-%d'), 
                      np.sum(self.FP.FRP), self.FP.nFires,
                      np.nansum(self.mapNoData > 0), np.nansum(self.mapDetectLim > 0),
                      np.nansum(self.mapArea == 0), self.grid.nx * self.grid.ny))
#        idxNonNan = np.isfinite(self.unified_map)
#        self.log.log('From unified map: missing %g, fires %g, below-detection %g, total FRP %g, mean detection limit %g' %
#                      (np.sum(np.isnan(self.unified_map)), 
#                      np.sum(self.unified_map[idxNonNan] > 0),
#                      np.nansum(self.unified_map[idxNonNan] < 0),
#                      np.nansum(self.unified_map[self.unified_map > 0]),
#                      np.nanmean(self.unified_map[self.unified_map < 0])))
        return


    #========================================================================

    def remove_bad_fires(self, idxBad):
        self.FP.FRP = np.delete(self.FP.FRP, idxBad)
        self.FP.lon = np.delete(self.FP.lon, idxBad)
        self.FP.lat = np.delete(self.FP.lat, idxBad)
        self.FP.dS = np.delete(self.FP.dS, idxBad)
        self.FP.dT = np.delete(self.FP.dT, idxBad)
        self.FP.T4 = np.delete(self.FP.T4, idxBad)
        self.FP.T4b = np.delete(self.FP.T4b, idxBad)
        self.FP.T11 = np.delete(self.FP.T11, idxBad)
        self.FP.T11b = np.delete(self.FP.T11b, idxBad)
        self.FP.TA = np.delete(self.FP.TA, idxBad)
        self.FP.ix = np.delete(self.FP.ix, idxBad)
        self.FP.iy = np.delete(self.FP.iy, idxBad)
        self.FP.time = np.delete(self.FP.time, idxBad)
        self.FP.line = np.delete(self.FP.line, idxBad)
        self.FP.sample = np.delete(self.FP.sample, idxBad)
        self.FP.SolZenAng = np.delete(self.FP.SolZenAng, idxBad)
        self.FP.ViewZenAng = np.delete(self.FP.ViewZenAng, idxBad)
        self.FP.LU = np.delete(self.FP.LU, idxBad)
        self.FP.satellite = np.delete(self.FP.satellite, idxBad)
        self.FP.nFires = len(self.FP.FRP)


    #========================================================================

    def plot_maps(self, chOutTemplate, ifHourly=False):
        #
        # Plots the map or a set of maps
        #
        bmap = basemap.Basemap(projection='cyl', resolution='f',    # crude, low, intermediate, high, full 
                               llcrnrlon = self.grid.x0, 
                               urcrnrlat = self.grid.y0 + self.grid.dy * (self.grid.ny-1),
                               urcrnrlon = self.grid.x0 + self.grid.dx * (self.grid.nx-1), 
                               llcrnrlat = self.grid.y0)
        #
        # Get the unified map to save drawings
        unified_map = self.make_unified_map()
        #
        # Now proceed with actual plotting
        #
        fig, ax = mpl.pyplot.subplots(1,1, figsize=(20,10))
        cmap = mpl.cm.get_cmap('jet')  #, lut)
        if ifHourly:
            for iHr in range(24):
                cs = bmap.pcolormesh(range(self.grid.nx), range(self.grid.ny), 
                                     unified_map[iHr,:,:], norm=None, cmap=cmap)
                cbar = bmap.colorbar(cs,location='bottom',pad="5%")
                cbar.set_label('FRP & detection-limit', fontsize=10)
                ax.set_title('Hourly FRP and -detection_limit', fontsize=12)
                # draw coastlines, state and country boundaries, edge of map.
                bmap.drawcoastlines(linewidth=0.5)
                bmap.drawcountries(linewidth=0.4)
                # draw parallels and meridians
                bmap.drawmeridians(np.arange(-180., 180., 30.), labels=[0,0,0,1],fontsize=10) 
                bmap.drawparallels(np.arange(-90., 90., 30.), labels=[1,0,0,0],fontsize=10)
                mpl.pyplot.savefig((self.day + iHr * spp.one_hour).strftime(chOutTemplate) + 
                                   '_' + self.gridName + '.png', dpi=300)
                mpl.pyplot.clf()
        else:
            # mapDay: any non-nan data overwrite nan
            # frp summed up, detection limit averaged
            mapDay = np.zeros(shape=(self.grid.nx, self.grid.ny))
            cntNoDetect = np.zeros(shape=(self.grid.nx, self.grid.ny))
            for iHr in range(24):
                mapDay += np.where(np.isfinite(unified_map[iHr,:,:]), 
                                   unified_map[iHr,:,:], 0.0)
                cntNoDetect[unified_map[iHr,:,:] < 0] += 1
            idxNoDetect = np.logical_and(cntNoDetect > 0, mapDay < 0)
            mapDay[idxNoDetect] /= cntNoDetect[idxNoDetect]
            # Draw
            print('colormesh...')
            
            cs = bmap.pcolormesh(range(self.grid.nx), range(self.grid.ny), mapDay, norm=None, cmap=cmap)
            cbar = bmap.colorbar(cs,location='bottom',pad="5%")
            cbar.set_label('FRP & detection-limit', fontsize=10)
            # draw coastlines, state and country boundaries, edge of map.
            bmap.drawcoastlines(linewidth=0.5)
            bmap.drawcountries(linewidth=0.4)
            # draw parallels and meridians
            bmap.drawmeridians(np.arange(-180., 180., 30.), labels=[0,0,0,1],fontsize=10) 
            bmap.drawparallels(np.arange(-90., 90., 30.), labels=[1,0,0,0],fontsize=10)
            ax.set_title('Hourly FRP and -detection_limit', fontsize=12)
            mpl.pyplot.savefig(self.day.strftime(chOutTemplate) + '_' + self.gridNames + 
                               '.png', dpi=300)
            mpl.pyplot.clf()
        mpl.pyplot.close()


    #========================================================================
    
    def UTC_vs_local_time(self, conversion_switch, dicOutMaps, ifBadDay):
        #
        # Distributes the filled-in maps of self (assumed UTC) to maps in local time 
        # Can do conversion in both directions
        #
        if conversion_switch == 'to_local_time':
            if self.ifLocalTime:
                self.log.log('Requested converson to local time but it is already local')
                raise ValueError
            dirSwitch = 1
            chTimeZone = 'LST'
        elif conversion_switch == 'to_UTC_time':
            if not self.ifLocalTime:
                self.log.log('Requested converson to UTC time but it is already UTC')
                raise ValueError
            dirSwitch = -1
            chTimeZone = 'UTC'
        else:
            self.log.log('Unknown switch: %s, can be to_local_time, to_UTC_time')
            raise ValueError
        #
        # check / create the 3-day environment: yesterday, today and tomorrow
        yesterday = self.day - spp.one_day
        tomorrow = self.day + spp.one_day
        for d in [yesterday, self.day, tomorrow]:
            try: dicOutMaps[d]
            except: 
                dicOutMaps[d] = daily_maps((self.grid, self.gridName), d, self.cloud_threshold,
                                           dirSwitch > 0, 24, self.LU, self.QA, self.log)
        #
        # grid can be rotated, have to make full case
        #
        localHr = np.zeros(shape=self.mapArea.shape, dtype=np.byte)
        ixs = np.ones(shape=(self.grid.ny,self.grid.nx)) * np.array(range(self.grid.nx))[None,:]
        iys = np.ones(shape=(self.grid.ny,self.grid.nx)) * np.array(range(self.grid.ny))[:,None]
        for iHr in range(24):
            localHr[iHr,:,:] = np.round(self.grid.grid_to_geo(ixs,iys)[0] / 
                                        15.0).astype(int) + iHr * dirSwitch
        for iHrFrom in range(24):
            for key, hrStart, hrEnd, hrDiff in [(yesterday,np.min(localHr[iHrFrom]), 0, 24), 
                                                (self.day, 0, 24, 0), 
                                                (tomorrow, 24, np.max(localHr[iHrFrom])+1, -24)]:
                for hrLST in range(hrStart, hrEnd):
                    idxToCopy = localHr[iHrFrom] == hrLST
                    if ifBadDay:
                        dicOutMaps[key].mapNoData[hrLST+hrDiff,:,:][idxToCopy] = 1.0
                        dicOutMaps[key].mapDetectLim[hrLST+hrDiff,:,:][idxToCopy] = 0.0  # absent data
                    else:
                        dicOutMaps[key].mapNoData[hrLST+hrDiff,:,:][idxToCopy] = self.mapNoData[iHrFrom,:,:][idxToCopy]
                        dicOutMaps[key].mapDetectLim[hrLST+hrDiff,:,:][idxToCopy] = self.mapDetectLim[iHrFrom,:,:][idxToCopy]
                    # pixel area is useful no matter what
                    dicOutMaps[key].mapArea[hrLST+hrDiff,:,:][idxToCopy] = self.mapArea[iHrFrom,:,:][idxToCopy]
        #
        # Now, distribute the fires. There might be none...
        #
        if self.FP.nFires > 0 and not ifBadDay:
#            print('Initial number of fires: ', self.FP.nFires)
            scale2hr = {'seconds':3600., 'hours':1.}[self.FP.chTimeStep]
#            FP_hrL = np.round(self.FP.time/3600. + dirSwitch * self.FP.lon / 15.0).astype(np.byte)
            FP_hrL = self.FP.time/scale2hr + dirSwitch * self.FP.lon / 15.0     # real number, in hours
        
            for mapD, idxD, hrDiff in [(dicOutMaps[yesterday], FP_hrL < 0, 24),
                                       (dicOutMaps[self.day], np.logical_and(FP_hrL>=0, FP_hrL<24), 0),
                                       (dicOutMaps[tomorrow], FP_hrL >= 24, -24)]:
                mapD.FP.FRP = np.append(mapD.FP.FRP, self.FP.FRP[idxD])
                mapD.FP.lon = np.append(mapD.FP.lon, self.FP.lon[idxD])
                mapD.FP.lat = np.append(mapD.FP.lat, self.FP.lat[idxD])
                mapD.FP.dS = np.append(mapD.FP.dS, self.FP.dS[idxD])
                mapD.FP.dT = np.append(mapD.FP.dT, self.FP.dT[idxD])
                mapD.FP.T4 = np.append(mapD.FP.T4, self.FP.T4[idxD])
                mapD.FP.T4b = np.append(mapD.FP.T4b, self.FP.T4b[idxD])
                mapD.FP.T11 = np.append(mapD.FP.T11, self.FP.T11[idxD])
                mapD.FP.T11b = np.append(mapD.FP.T11b, self.FP.T11b[idxD])
                mapD.FP.TA = np.append(mapD.FP.TA ,self.FP.TA[idxD])
                mapD.FP.ix = np.append(mapD.FP.ix, self.FP.ix[idxD])
                mapD.FP.iy = np.append(mapD.FP.iy, self.FP.iy[idxD])
                mapD.FP.time = np.append(mapD.FP.time, np.round((FP_hrL[idxD] + hrDiff)*scale2hr).astype(np.int64)) # Local time now
                mapD.FP.time[mapD.FP.time == 24*scale2hr] = 24*scale2hr - 1  # rounding may bring it to 00:00 of next day, subtract 1 second
                mapD.FP.line = np.append(mapD.FP.line, self.FP.line[idxD])
                mapD.FP.sample = np.append(mapD.FP.sample, self.FP.sample[idxD])
                mapD.FP.SolZenAng = np.append(mapD.FP.SolZenAng, self.FP.SolZenAng[idxD])
                mapD.FP.ViewZenAng = np.append(mapD.FP.ViewZenAng, self.FP.ViewZenAng[idxD])
                mapD.FP.LU = np.append(mapD.FP.LU, self.FP.LU[idxD])
                mapD.FP.satellite = np.append(mapD.FP.satellite, self.FP.satellite[idxD])
                mapD.FP.nFires = len(mapD.FP.FRP)
                if mapD.FP.timeStart is None:
                    mapD.FP.timeStart = self.FP.timeStart - spp.one_hour * hrDiff
                    mapD.FP.timezone = chTimeZone
                    mapD.FP.chTimeStep = self.FP.chTimeStep
                else:
                    if mapD.FP.timeStart != self.FP.timeStart - spp.one_hour * hrDiff:
                        raise ValueError('Time start does not correspond. \nMine:' + str(self.FP.timeStart) + ', shift=' + str(spp.one_hour * hrDiff) 
                                         + '\nmapD time start: ' + str(mapD.FP.timeStart))
                # stupidity check
                if np.any(mapD.FP.time < 0) or np.any(mapD.FP.time >= 24*scale2hr):
                    print(mapD.FP.time, '\nmax_time, argmax_time, sum(time>=24*3600), wrong_lon, wrong_lat\n', 
                          np.max(mapD.FP.time), np.argmax(mapD.FP.time), 
                          np.sum(mapD.FP.time >= 24*scale2hr), mapD.FP.lon[np.argmax(mapD.FP.time)], 
                          mapD.FP.lat[np.argmax(mapD.FP.time)])
                    raise ValueError
                
#                print('Nbr of fires: in', hrDiff, mapD.FP.nFires)


    #=========================================================================

    def hourly_to_daily_single_map(self, mapDiurnalVars): #, ifBadDay):
        #
        # Creates a daily field from the hourly ones and adds to the current map.
        # The meaning of the unified daily map is:
        # - nan means no observations during that day
        # - negative means the detection limit for daily-mean FRP, accounting for diurnal cycle
        # - positive value is the daily-mean FRP accounting for the daily cycle 
        #
        # Detection limit. Daily limit is such that no observations during that day
        # can see the fire. For cloudy day it is plus-infinity, of course, but 
        # clear sky day puts the following limit 
        # D_day = min_over_hr (D_hr / Iemis_hr)
        # D is detection limit for the day / hour, Iemis is emission diurnal variation
        # for the specific land use
        #
        # detection limit is zero for non-observed areas and finite for actual data
        # cloud threshold is needed: cloudy pixels mean infinitely high detection limit
        #
#        print('Calling............')
#        self.detectLim_clds_daily = np.nanmin(np.where
#                                         (np.logical_and(self.mapDetectLim * mapDiurnalVars > 0,
#                                                         self.mapNoData <= self.cloud_threshold),
#                                          self.mapDetectLim / (mapDiurnalVars + 1e-15), 
#                                          np.nan), axis=0)
        # What-if map: detection limit for clear sky
        detectLim_clrSky_hourly = np.where(self.mapDetectLim * mapDiurnalVars > 0,
                                           self.mapDetectLim / (mapDiurnalVars + 1e-15), np.nan)
        # Accounting for cloud coverage with sigmoid function. Still hourly
        detectLim_clds_hourly = detectLim_clrSky_hourly * (1.+ self.cld_sigmoid_scale / 
                                                           (1.+ np.exp(-self.cld_sigmoid_slope * 
                                                                       (self.mapNoData - 
                                                                        self.cld_sigmoid_shift))))
        # What-if map: detection limit for clear sky
        # see notebook 12, pp.1-4
        #
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        self.detectLim_clrSky_daily = np.nanmin(detectLim_clrSky_hourly, axis=0)
        self.detectLim_clds_daily = np.nanmin(detectLim_clds_hourly, axis=0)
        warnings.filterwarnings('default', r'All-NaN slice encountered')
        
#        # Daytime detection limit: clouds or no overpass make it something very big, e.g. 1000 MW
#        # For partial clouds, the size-decided detection limit is scaled towards
#        # 1000 MW, which is reaches when cloud fraction reaches threshold 
#        detectLim_daytime = np.where(self.mapNoData < self.cloud_threshold * 0.999,
#                                     np.nanmin(np.where(self.mapDetectLim[11:14,:,:] == 0,
#                                                        1000., 
#                                                        self.mapDetectLim[11:14,:,:]), axis=0) /
#                                     (1.- self.mapNoData / self.cloud_threshold),
#                                     1000.)
        #
        # Non-observed and sunglint areas: 100% loss, clouds: loss of cloud_fraction data
        # from 0 to 1, no need in threshold here
        self.NoData_daily = np.min(self.mapNoData, axis=0)
#        self.NoData_daily[np.all(self.mapNoData > self.cloud_threshold, axis=0)] = np.nan
        #
        # For fires, create a set of vectors showing LU, daily mean and daily variation
        # all for projected map.
        #
#        if ifBadDay: self.FP.nFires = 0
            
        if self.FP.nFires > 0:
            det_lim_lst = detectLim_clds_hourly[:,self.FP.iy,self.FP.ix]    # clear-sky
            det_lim_lst[det_lim_lst == 0] = 1e6
#            missing_lst = self.mapNoData[:,self.FP_iy,self.FP_ix]
            #
            # Collapse individual fires to the gridded list (nLU, nFiresGridded)
            #
            FP_hour = np.round(self.FP.time / 3600).astype(int)

            FRP_lst = pixel_projection.fires_2_unified_map(FP_hour, self.FP.LU, self.FP.ix,
                                                           self.FP.iy, self.FP.FRP, self.LU.diurnal,
                                                           det_lim_lst, 
                                                           self.obsNightHrs, self.obsDayHrs,
                                                           self.obsNightHrs.shape[0], 
                                                           self.obsDayHrs.shape[0],
                                                           len(self.LU.LUtypes), self.FP.nFires)
            # Output is an array 27 x nGriddedFires. 0:4 = ix, iy, LU, FRP, frp_diurnal[0:23]
            #
            self.nGriddedFires = np.sum(FRP_lst[3,:] > 0)
            self.FPgrid_ix = np.round(FRP_lst[0,:self.nGriddedFires]).astype(np.int32)
            self.FPgrid_iy = np.round(FRP_lst[1,:self.nGriddedFires]).astype(np.int32)
            self.FPgrid_LU = np.round(FRP_lst[2,:self.nGriddedFires]).astype(np.int32)
            self.FPgrid_frp = FRP_lst[3,:self.nGriddedFires]
            # Transpose this guy: in Python, last index changes fastest
            self.FPgrid_diurnal = FRP_lst[4:28,:self.nGriddedFires].T
#            print('Fires: ', self.nFires, 'Gridded fires:', nGriddedFires)
#            print('FRP mean ',self.FRPmean_lst,'\n FRP ratio',self.FRPratio_lst)

            # A summary.
#            self.log.log(self.day.strftime('%Y%m%d'))
            # Land uses with non-zero day and night
#            print(idxLU_nonzero, idxLU_nonzero.shape)
#            for iLU, chLU in enumerate(self.LU.LUtypes): #[idxLU_nonzero]):
#                idxLU = self.FPgrid_LU == iLU
#                if np.sum(idxLU) > 0:
#                    self.log.log('Ratio_FRP_night_vs_day: %s %s' % 
#                                 (chLU, str(np.mean(self.FPgrid_diurnal[idxLU][:,self.obsDayHrs]) /
#                                           np.mean(self.FPgrid_diurnal[idxLU][:,self.obsNightHrs]))))
        else: 
            self.nGriddedFires = 0


    #========================================================================

    def update_land_use_hourly_file(self, newLU):
        #
        # If we happen to get new land use, fires need to be reprojected to it. This is just to avoid
        # messing with MODIS raw data. We need to replace the land_use object with new and project 
        # fires to the new set of classes. 
        #
        if self.FP.nFires > 0:
            self.FP.LU = newLU.get_LU_4_fires(self.FP.lon, self.FP.lat)

        self.LU = newLU
        

    #========================================================================
        
    def make_unified_daily_map(self, idxLandUse, timer=None):
        #
        # Make the unified map, possibly, for the requested land use
        # Works with DAILY data and produces DAILY unified map.
        # At hourly level, we do not need unified map - one can sum-up components if desires
        #
        # Start from zero-map
#        print(self.day)
        try: self.unified_map_base.shape
        except:
            self.unified_map_base = self.detectLim_clrSky_daily * 0.0
    #        np.zeros(shape=self.detectLim_clds_daily.shape) * self.detectLim_clrSky_daily
#            print('shape: finite cells: ',np.sum(np.isfinite(self.unified_map_base)), 
#                  'nansum', np.nansum(self.unified_map_base))
                # clouds etc: no data
            if timer is not None: timer.start_timer('no_data')
            self.unified_map_base[self.NoData_daily > self.cloud_threshold] = np.nan
            if timer is not None: timer.stop_timer('no_data')
#            print('Nodata applied: finite cells: ',np.sum(np.isfinite(self.unified_map_base)), 
#                  'nansum', np.nansum(self.unified_map_base), 'nanmax', np.nanmax(self.unified_map_base))
            # Unified map is -detection_limit if no other suggestions
            # in the end, these will be reported zeroes: put the minus-detection-limit
            if timer is not None: timer.start_timer('detection_limit')
            self.unified_map_base[self.unified_map_base == 0] = - self.detectLim_clds_daily[self.unified_map_base == 0]
            if timer is not None: timer.stop_timer('detection_limit')
#            print('Detection limit applied: finite cells: ',np.sum(np.isfinite(self.unified_map_base)), 
#                  'nansum', np.nansum(self.unified_map_base), 'nanmax', np.nanmax(self.unified_map_base))
        if timer is not None: timer.start_timer('allocation_and_fires')
        self.unified_map = self.unified_map_base.copy()
#        print('Unified_map ready for FRP: finite cells: ',np.sum(np.isfinite(self.unified_map)), 
#              'nansum', np.nansum(self.unified_map), 'nanmax', np.nanmax(self.unified_map))
        #
        # If there are fires, add
        if self.FP.nFires > 0:
            if idxLandUse == len(self.LU.LUtypes):   #'LU_all':
                self.unified_map[self.FPgrid_iy, self.FPgrid_ix] = self.FPgrid_frp
            else:
                idxOK = self.FPgrid_LU == idxLandUse
#                print('Cells foumd:', np.sum(idxOK))
                self.unified_map[self.FPgrid_iy[idxOK], 
                                 self.FPgrid_ix[idxOK]] = self.FPgrid_frp[idxOK]
        if timer is not None: timer.stop_timer('allocation_and_fires')


    #=======================================================================
    
    def to_file(self, chOutFNm, ifDailyMaps=False):
        #
        # Stores tje produced daily map into the netCDF file. Stores a single file per day
        # with either daily fields or a time series of 24 hourly fields
        #
        if ifDailyMaps:
#            vars2d = ['unified_IS4FIRES_v3_0_daily']
#            maps = {'unified_IS4FIRES_v3_0_daily':self.unified_daily_map[None,:,:]}
            vars2d = ['detection_limit_cld_daily','missing_cells_daily',
                      'detection_limit_clearsky_daily']
            #
            # This would set it to the middle of the day treating daily-mean as ave(00:00 - 23:00)
#            today = dt.datetime(self.day.year, self.day.month, self.day.day, 11, 30)
            #
            # And this is to end of the day - actually, 00:00 of the next day
#            today = dt.datetime(self.day.year, self.day.month, self.day.day) + spp.one_day
#            arTimes = [today]
            #
            # And this is just the day itself, actually, the beginning of the day. For fire redcords is fine
            arTimes = [self.day]
            maps = {'detection_limit_cld_daily' : self.detectLim_clds_daily[None,:,:], 
                    'missing_cells_daily' : self.NoData_daily[None,:,:],
                    'detection_limit_clearsky_daily' : self.detectLim_clrSky_daily[None,:,:]}
        else:
            vars2d = [#'unified_IS4FIRES_v3_0', 
                      'Pixels_tot_area', 'detection_limit', 'missing_cells']
            arTimes = list((self.day + i*spp.one_hour for i in range(24)))
            maps = {#'unified_IS4FIRES_v3_0':self.unified_map,
                    'Pixels_tot_area':self.mapArea, 'detection_limit':self.mapDetectLim,
                    'missing_cells':self.mapNoData}
        #
        # Universal dictionaries
        units = {'unified_IS4FIRES_v3_0':'undefined', 'unified_IS4FIRES_v3_0_daily':'undefined',
                 'Pixels_tot_area':'km2', 'detection_limit':'MW', 'missing_cells':'',
                 'detection_limit_cld_daily':'MW', 'missing_cells_daily':'',
                 'detection_limit_clearsky_daily':'MW'}
        #
        # Output file will the normal SILAM file
        # Careful! 
        # For hourly field, 00-23 are in the daily file, for daily field it will be 00 of the next day
        #
        outF = silamfile.open_ncF_out(chOutFNm,  # self.day.strftime(chOutTemplate), # output file name
                                      'NETCDF4',   # nctype, 
                                      self.grid,
                                      silamfile.SilamSurfaceVertical(), 
                                      self.day,    # anTime, 
                                      arTimes,    # arrTime, v
                                      [],          # vars3d, 
                                      vars2d,      # vars2d, 
                                      units, 
                                      -999.,       # fill_value, 
                                      True,        # ifCompress, 
                                      2, None,  # ppc, hst
                                      timezone = 'LST' if self.ifLocalTime else 'UTC')  # timezone to be written in nc
        #
        # Grid group is written in silamfile
        silamfile.write_grid_to_nc(outF, self.grid, self.gridName)
        outF.grid_projection = 'lonlat'
        #
        # A bit of metadata and land_use types
        outF.cloud_threshold = self.cloud_threshold
        outF.nFires = self.FP.nFires
        if ifDailyMaps: outF.nGriddedFires = self.nGriddedFires
        outF.ifLocalTime = {True:'True',False:'False'}[self.ifLocalTime]
        # LU
        self.LU.to_nc_file(outF)
        # Fire records
        if self.FP.nFires > 0:
            #
            # Fire records require land_use and QA metadata, but while in map object these may be
            # present only at the map level
            #
            self.FP.LU_metadata = self.LU.metadataFNm
            self.FP.QA = self.QA
            #
            # Fire records - use their native writer
            #
            self.FP.to_nc(outF)      # Raw fires as observed by MODIS...
            #
            # ... plus gridded daily fires
            #
            if ifDailyMaps:
                firesGridAxis = outF.createDimension("gridded_fires", self.nGriddedFires)
                hrsAxis = outF.createDimension("hours_LST", 24)
                #                   var_name, type, long name, unit
                for FP_dayvar in [('FPgrid_frp','f4','FRP','MW', self.FPgrid_frp),
#                                  ('FPgrid_frp_ratio','f4','FRP_min_max_ratio','', self.FPgrid_ratio),
                                  ('FPgrid_LU','i2','grid land_use','', self.FPgrid_LU),
                                  ('FPgrid_ix','i2','grid x-index','', self.FPgrid_ix),
                                  ('FPgrid_iy','i2','grid y-index','', self.FPgrid_iy)]:
                    vFPg = outF.createVariable(FP_dayvar[0], FP_dayvar[1], ("gridded_fires"), 
                                               zlib=True, complevel=5) #least_significant_digit=5)
                    outF.variables[FP_dayvar[0]][:] = FP_dayvar[4][:]
                    vFPg.long_name = FP_dayvar[2]
                    if FP_dayvar[3] != '': vFPg.units = FP_dayvar[3]
                #
                # And the diurnal variation fitted for this specific grid cell
                #
                vFPg = outF.createVariable('FPgrid_diurnalvar', 'f4', ("gridded_fires", "hours_LST"), 
                                               zlib=True, complevel=5) #least_significant_digit=5)
                outF.variables['FPgrid_diurnalvar'][:] = self.FPgrid_diurnal[:,:]
                vFPg.long_name = 'realtive FRP at hour of day, LST'
                vFPg.units = ''
                
        else:
            self.log.log(self.day.strftime('No fires for %Y%m%d'))
        #
        # Store the map data
        #
        for v in vars2d:
            outF.variables[v][:,:,:] = maps[v][:,:,:]
        #
        # Store the QA object
        #
        self.QA.to_nc(outF)
        # done
        outF.close()



########################################################################################################
########################################################################################################
#
# Generic functions using the daily_maps
#
########################################################################################################
########################################################################################################

def check_file(chFNm):

    a = np.ones(shape=(10))
    b = np.ones(shape=(10)) * 2.0

    c = zip(a,b)
    print(list(set(c)))
    cset = np.array(list(set(zip(a,b))))
    print(cset, cset[0])
    
#    for c in zip(a,b): print(c)
#    sys.exit()
    
    fIn = nc4.Dataset(chFNm,'r')
    hr = fIn.variables['FP_hour'][:]
    lon = fIn.variables['FP_lon'][:]
    lat = fIn.variables['FP_lat'][:]
    print(hr, np.max(hr), np.argmax(hr), np.sum(hr > 23), lon[np.argmax(hr)], lat[np.argmax(hr)])
        
#############################################################################

def find_duplicate_files(tStart, tEnd, tStep, MxD14_templ, MxD35_templ, chJunkDir):
    #
    # Goes through the given time period checking that granules are not duplicated.
    # In case of a duplicate, finds the file with the latest processing date and checks that
    # it is readable. If yes, deletes (relocates) the other files
    #
    now = tStart
    log = spp.log('run_duplicate_files.log')
    #
    # Go through the day
    while now < tEnd:
        print(now.strftime('%Y%m%d-%H%M'))
        #
        # Cycle over the available satellites
        for satellite in ['MOD','MYD']:
            #
            # Check the tempalte: if only one or no files, take the next
            #
            if not path.exists(path.join(chJunkDir, satellite)): 
                os.makedirs(path.join(chJunkDir, satellite))
            arFNms14 = sorted(glob.glob(now.strftime(MxD14_templ.replace('MxD',satellite))))
            arFNms35 = sorted(glob.glob(now.strftime(MxD35_templ.replace('MxD',satellite))))
            print(now.strftime(MxD14_templ), arFNms14, arFNms35)
            if len(arFNms14) <= 1 and len(arFNms35) <= 1: continue
            # Duplicates.
            print('Several files satisfy to (one of) templates', arFNms14, arFNms35)
            i14_good = -1
            i35_good = -1
            for i14 in range(len(arFNms14)-1, -1, -1):
                for i35 in range(len(arFNms35)-1, -1, -1):
                    gran = sgMOD.granule_MODIS(satellite, now, arFNms14[i14], arFNms35[i35], log)
                    # get the data
                    if gran.pick_granule_data_IS4FIRES_v3_0(): 
                        i14_good = i14
                        i35_good = i35
                        arFNms14.pop(i14)
                        arFNms35.pop(i35)
                        break
                    else: log.log('Unreadable granule: %s, %s' % 
                                  (path.split(now.strftime(MxD14_templ).replace('MxD',satellite))[1],
                                   path.split(now.strftime(MxD35_templ).replace('MxD',satellite))[1]))
            if i14_good < 0 or i35_good < 0:
                log.log(now.strftime('Cannot find useful pair for this time moment: %Y%m%d_%H%M'))
            # Move duplicates to a separate directory
            for FNm in arFNms14 + arFNms35:
                shutil.move(FNm, path.join(chJunkDir, satellite))
            
        now += tStep


#=======================================================================

def count_active_fire_cells(dayStart, dayEnd, ifSplitLU, chFNmtemplate, ifMakeTSM, log, 
                            first_good_file_timestamp, mpirank, mpisize, communicator):
    #
    # Reads through the given period of daily maps and returns the map
    # of non-zero fires
    # dayStart and dayEnd only designate the period. The actual timing is taken from the files
    # Reason: daily files refer to the end of the averaging period, i.e. 00:00 of the next day
    #
    MapWrk = daily_maps(log=log)
    #
    # Attention! dayStart is given, but the time tag is the next day 00:00.
    #
    MapWrk.from_file((dt.datetime(dayStart.year,dayStart.month,dayStart.day) + spp.one_day).strftime(chFNmtemplate))
    
    nDays = (dayEnd - dayStart).days    # not beyond that: these are just days, not actual times
    timeStart = MapWrk.day   #  mid-day of the dayStart
    today = MapWrk.day   #  mid-day of the dayStart
    tmpDir = os.path.join(os.path.split(chFNmtemplate)[0],'tmp')
    spp.ensure_directory_MPI(tmpDir, spp.one_minute * 5.0)
    
    ifMakePickle = False
    chPickleFNm = 'map_frp_times_LU_split.pickle' if ifSplitLU else 'map_frp_times_LU_all.pickle'
    
    if mpirank == 0 and ifMakePickle:
        if ifSplitLU:
            # Every land use in a separate map
            map_frp = np.zeros(shape = (len(MapWrk.LU.LUtypes) + 1, MapWrk.grid.ny, MapWrk.grid.nx), 
                                 dtype=np.float32)
            validTimes = np.zeros(shape = (nDays, len(MapWrk.LU.LUtypes) + 1), 
                                  dtype=bool)
            iT = 0
            iBad = 0
            while today < dayEnd:
                MapWrk.from_file(today.strftime(chFNmtemplate))
                if MapWrk.FP.nFires > 0:
                    map_frp[MapWrk.FPgrid_LU, MapWrk.FPgrid_iy, MapWrk.FPgrid_ix] += MapWrk.FPgrid_frp
                    validTimes[iT,np.array(list(set(MapWrk.FPgrid_LU)))] = True
                    if np.min(MapWrk.FP.FRP) < 1e-5:
                        iMinFire = np.argmin(MapWrk.FP.FRP)
                        iBad += 1
                        print(today, np.min(MapWrk.FP.FRP), 'at', iMinFire)
                        print('MIN fire: lon=%g, lat=%g, FRP=%g, iLU=%i, LU=%s, satellite=%s, time=%s' %
                              (MapWrk.FP.lon[iMinFire], MapWrk.FP.lat[iMinFire], MapWrk.FP.FRP[iMinFire],
                               MapWrk.FP.LU[iMinFire], MapWrk.LU.LUtypes[MapWrk.FP.LU[iMinFire]], 
                               MapWrk.FP.satellite[iMinFire], 
                               (MapWrk.FP.time[iMinFire] * spp.one_second + MapWrk.FP.timeStart).strftime('%Y%m%d-%H%M-jday%j LST')))
                    if np.max(MapWrk.FP.FRP) > 1e10:
                        iMaxFire = np.argmax(MapWrk.FP.FRP)
                        iBad += 1
                        print(today, np.max(MapWrk.FP.FRP), 'at', iMaxFire)
                        print('MAX fire: lon=%g, lat=%g, FRP=%g, iLU=%i, LU=%s, satellite=%s, time=%s' %
                              (MapWrk.FP.lon[iMaxFire], MapWrk.FP.lat[iMaxFire], MapWrk.FP.FRP[iMaxFire],
                               MapWrk.FP.LU[iMaxFire], MapWrk.LU.LUtypes[MapWrk.FP.LU[iMaxFire]],
                               MapWrk.FP.satellite[iMaxFire],
                               str(MapWrk.FP.time[iMaxFire] * spp.one_second + MapWrk.FP.timeStart)))
                today += spp.one_day
                iT += 1
                if today.day == 15 and today.month == 1: print(today)
            if iBad > 0:
                print('Number of small / huge fires:', iBad)
            # now, make total: all LUs togeher
            map_frp[-1,:,:] = np.sum(map_frp[:-1,:,:], axis=0)
            validTimes[:,-1] = np.any(validTimes[:,:-1], axis=1)
            print('No LU_all')
#            LU_names_loc = np.append(MapWrk.LU.LUtypes, np.array(['LU_all']))
            LU_names_loc = MapWrk.LU.LUtypes
        else:
            # All land uses in a single map
            map_frp = np.zeros(shape = (1, MapWrk.grid.ny, MapWrk.grid.nx), dtype=np.float32)
            validTimes = np.zeros(shape = (nDays, 1), dtype=bool)
            iT = 0
            while today < dayEnd:
                MapWrk.from_file(today.strftime(chFNmtemplate))
                if MapWrk.FP.nFires > 0:
                    map_frp[0, MapWrk.FPgrid_iy, MapWrk.FPgrid_ix] += MapWrk.FPgrid_frp
                    validTimes[iT,0] = True
                today += spp.one_day
                iT += 1
                if today.day == 15 and today.month == 1: print(today)
            LU_names_loc = np.array(['LU_all'])
        # summary
        totals_from_map = {}
        for iLU, LU in enumerate(LU_names_loc):
            totals_from_map[LU] = np.nansum(map_frp[iLU,:,:])
            log.log('Total from map_frp, valid times and cells for %s %s: %g %g %g' % 
                    (MapWrk.gridName, LU, np.nansum(map_frp[iLU,:,:]), np.sum(validTimes[:,iLU]), 
                     np.sum(map_frp[iLU,:,:]>0.0)))
        #
        # Store the map and valid times into pickle for all processes to read it
        if ifMakeTSM:
            with open(os.path.join(tmpDir,chPickleFNm), 'wb') as handle:
                pickle.dump((map_frp, validTimes, LU_names_loc), 
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(mpirank, 'pickle ready ', MapWrk.gridName)

    # non-0 MPIs wait here
    spp.MPI_join('count_active_fire_cells', tmpDir, mpisize, mpirank, communicator, spp.one_hour * 2.0, log)
    log.log('%03i after waiting' % mpirank)
    #
    # If tsMatrices are requested to the output, have to run through the time series again
    # Hopefully, they are buffered
    #
    if ifMakeTSM:
        if mpirank != 0 or not ifMakePickle:
            print(mpirank, 'opening pickle')
            with open(os.path.join(tmpDir,chPickleFNm), 'rb') as handle:
                map_frp, validTimes, LU_names_loc = pickle.load(handle)
            print(mpirank, 'consumed')
        timesLst = list((timeStart + iDay * spp.one_day for iDay in range(nDays)))
        times = np.array(timesLst)
        for iLU, LU in enumerate(LU_names_loc):
            
#            if LU != 'AF_grass':
#                print('AF_grass is being debugged, not: ', LU) 
#                continue
            
            if np.mod(iLU, mpisize) != mpirank: continue
            #
            # Define the output file name
            if '_map_' in chFNmtemplate:
                chFNmOutTmp = chFNmtemplate.replace('_map_', '_tsMatrix_%s_' % LU)
            else:
                chFNmOutTmp = chFNmtemplate.replace('.nc','_tsMatrix_%s.nc' % LU)
            # tsMatrices are small enough to have all days at once, no time templates
            chFNmOut = chFNmOutTmp.replace('%Y','').replace('%m','').replace('%d','')
            # good file exists?
            if os.path.exists(chFNmOut):
                if os.stat(chFNmOut).st_mtime > first_good_file_timestamp:
                    log.log('...no need to do anything for %s: output OK' % chFNmOut) 
                    continue
            #
            # The non-zero grid cells
            valid_cells_tmp = np.argwhere(map_frp[iLU,:,:] > 0.0)   # indices valid cells, # iy,ix note the order
            if len(valid_cells_tmp) == 0:
                print('No active fires for ', LU)
                continue
            else:
                print('Active cells,  times', LU, len(valid_cells_tmp), np.sum(validTimes[:,iLU]))
            valid_geo = MapWrk.grid.grid_to_geo(valid_cells_tmp[:,1], valid_cells_tmp[:,0])  # iy,ix note the order
            cells_tmp = list( (stations.Station('%03g_%03g' % (valid_cells_tmp[i][1],
                                                               valid_cells_tmp[i][0]),
                                                '%03g_%03g' % (valid_cells_tmp[i][1],
                                                               valid_cells_tmp[i][0]), 
                                                valid_geo[0][i], valid_geo[1][i])
                                                for i in range(valid_cells_tmp.shape[0])) )
            argsort_cells = np.argsort(cells_tmp)
            cells = np.array(cells_tmp)[argsort_cells]
            valid_cells = valid_cells_tmp[argsort_cells]
            TSM_cells_idx = np.ones(shape=(map_frp.shape[1], map_frp.shape[2]), 
                                    dtype=np.int32) * (-999)
            TSM_cells_idx[valid_cells[:,0], valid_cells[:,1]] = range(valid_cells.shape[0])
            #
            # A trick: LU_all cannot be filled in vector form because several LUs can be found in 
            # a single grid cell - then the vector notation leads to the last one overwriting
            # the previous ones even if += operation is used. Therefore, we store the empty tsm 
            # and fill it in gradually while looking at other LUs
            #
#            if LU == 'LU_all':
#                tsmVals_LUa = np.zeros(shape=(times.shape[0], valid_cells.shape[0])) #* np.nan
#                TSM_cells_idx_LUa = copy.deepcopy(TSM_cells_idx)
#                cells_LUa = copy.deepcopy(cells)
#                chFNmOut_LUa = copy.deepcopy(chFNmOut)
#                continue
#            else:

            tsmVals = np.zeros(shape=(times.shape[0], valid_cells.shape[0])) * np.nan
            #
            # Scan the time period
            for today in times[validTimes[:,iLU]]:
                iT = (today - timeStart).days
                # get next daily map
                MapWrk.from_file(today.strftime(chFNmtemplate))
                if LU == 'LU_all': iLU_tmp = -1
                else: iLU_tmp = iLU
                if MapWrk.FP.nFires > 0: 
                    tsmVals[iT,:] = pixel_projection.collect_frp_cells(iLU_tmp, MapWrk.FPgrid_LU, 
                                                                       MapWrk.FPgrid_ix, 
                                                                       MapWrk.FPgrid_iy, 
                                                                       MapWrk.FPgrid_frp, 
                                                                       TSM_cells_idx,
                                                                       MapWrk.detectLim_clds_daily, 
                                                                       len(cells), 
                                                                       len(MapWrk.FPgrid_frp),
                                                                       map_frp.shape[2],  # nx 
                                                                       map_frp.shape[1])  # ny

#(iLU, FPgrid_LU, FPgrid_ix, FPgrid_iy, FPgrid_frp, & 
#                           & TSM_cells_idx, tsmVals, nCells, nFires, nx, ny                    
#                    
#                    idxLUOK = MapWrk.FPgrid_LU == iLU
#                    if np.sum(idxLUOK) == 0: continue
#                    tsmVals[iT, TSM_cells_idx[MapWrk.FPgrid_iy[idxLUOK], 
#                                              MapWrk.FPgrid_ix[idxLUOK]]] = MapWrk.FPgrid_frp[idxLUOK]
##                    tsmVals_LUa[iT,
##                                TSM_cells_idx_LUa[MapWrk.FPgrid_iy[idxLUOK], 
##                                                  MapWrk.FPgrid_ix[idxLUOK]]] += MapWrk.FPgrid_frp[idxLUOK]

            TSM = MyTimeVars.TsMatrix(timesLst, cells, ['FRP'], tsmVals, ['MW'], -999999)
            TSM.to_nc(chFNmOut)
            log.log('Totals from TSM of %s, %s: %g' % (MapWrk.gridName, LU, 
                                                       np.nansum(tsmVals[tsmVals>0])))
#        # And also LU_all
#        LU = 'LU_all'
#        tsmVals_LUa[tsmVals_LUa == 0] = np.nan
#        TSM = MyTimeVars.TsMatrix(timesLst, cells_LUa, ['FRP'], tsmVals_LUa, ['MW'], -999999)
#        TSM.to_nc(chFNmOut_LUa)
#        log.log('Totals from map and TSM of %s: %g, %g' % 
#                (LU, totals_from_map[LU], np.nansum(tsmVals_LUa)))
    # wait here
    spp.MPI_join('count_active_fire_cells_2', tmpDir, mpisize, mpirank, communicator, spp.one_hour * 5, log)
    return map_frp


##############################################################################

def count_missing_pixels_fraction(dayStart, dayEnd, chFNmtemplate, nBurn_pixels, iMonth, outDir, log):
    #
    # Goes over the given time period and computes mean fraction of missing pixels
    # Can account for only given month and availability of observed fires nearby
    #
    MapWrk = daily_maps(log=log)
    MapWrk.from_file(dayStart.strftime(chFNmtemplate))
    nDays = (dayEnd - dayStart).days    # not beyond that: these are just days, not actual times
    today = MapWrk.day   #  mid-day of the dayStart
    fMissing = -999.0
    map_miss = np.ones(shape=(MapWrk.grid.ny,MapWrk.grid.nx),dtype=np.float32) * fMissing
    map_cnt = np.zeros(shape=(MapWrk.grid.ny,MapWrk.grid.nx),dtype=np.int32)
    outF = silamfile.open_ncF_out(os.path.join(outDir,'missing_cells_fract_%s_rad%g.nc4' % 
                                               (MapWrk.gridName, nBurn_pixels)),
                                  'NETCDF4', MapWrk.grid, silamfile.SilamSurfaceVertical(), 
                                  today, list((today + i * spp.one_day for i in range(nDays))), [], 
                                  ['missing_cells_daily'], 
                                  {'missing_cells_daily':''},
                                  fMissing, True, 3, None)
    iT = 0
    while today < dayEnd:
        map_frp = np.zeros(shape = (MapWrk.grid.ny, MapWrk.grid.nx), dtype=np.float32)
        map_missTmp = np.ones(shape=(MapWrk.grid.ny,MapWrk.grid.nx),dtype=np.float32) * fMissing
        #
        # Needed month?
        if iMonth < 0 or today.month == iMonth:
            #
            # Process daily map: read, find burning areas and store only them
            #
            MapWrk.from_file(today.strftime(chFNmtemplate))
            if MapWrk.FP.nFires > 0:
                if nBurn_pixels > 0:
                    map_frp[MapWrk.FPgrid_iy, MapWrk.FPgrid_ix] += MapWrk.FPgrid_frp
                    map_missTmp[:,:] = pixel_projection.select_burning_regions(
                                                          map_frp, MapWrk.NoData_daily, nBurn_pixels,
                                                          fMissing, MapWrk.grid.nx, MapWrk.grid.ny)
                else:
                    map_missTmp = MapWrk.NoData_daily
                #
                # Count the values over the fire regions
                #
                idxOK = map_missTmp > (fMissing + 1)
                idxClean = np.logical_and(idxOK, map_miss < (fMissing + 1))
                map_miss[idxClean] = 0
                map_miss[idxOK] += map_missTmp[idxOK]
                map_cnt[idxOK] += 1
        today += spp.one_day
        iT += 1
        if today.day == 15 and today.month == 1: print(today)
        #
        # Store the daily map
        #
        outF['missing_cells_daily'][iT,:,:] = map_missTmp[:,:]
    outF.close()
    #
    # Now, store the mean fields
    # 
    idxOK = map_cnt > 0
    map_miss[idxOK] /= map_cnt[idxOK]
    outF = silamfile.open_ncF_out(os.path.join(outDir,'missing_cells_fract_mean_%s_rad_%g_%s_%s.nc4' %
                                               (MapWrk.gridName, nBurn_pixels, dayStart.strftime('%Y%m%d'),
                                                dayEnd.strftime('%Y%m%d'))),
                                  'NETCDF4', MapWrk.grid, silamfile.SilamSurfaceVertical(), 
                                  today, [today], [], 
                                  ['missing_cells_mean'], 
                                  {'missing_cells_mean':''},
                                  fMissing, True, 3, None)
    outF['missing_cells_mean'][0,:,:] = map_miss[:,:]
    outF.close()
    
    return map_miss


#=======================================================================

def get_day_night_ratio(ifReadRawData, ifAnalyseData, land_use, chInTemplate, tStart, tEnd, 
                        chOutDir, log):
    #
    # Scan the time period and compute the ratio between day and night
    #
    if ifReadRawData:
        MapWrk = daily_maps(log=log)
        today  = tStart
        fpFRPlst = []  #np.array([], dtype = np.float32)
        fpLSThourlst = []  #np.array([], dtype = np.float32)
        fpLonlst = []  #np.array([], dtype = np.float32)
        fpLatlst = []  #np.array([], dtype = np.float32)
        dayslst = []
        #
        # go over time
        nFires = 0
        while today < tEnd:
            MapWrk.from_file(today.strftime(chInTemplate))
            if MapWrk.FP.nFires > 0:
                fpFRPlst.append(MapWrk.FP_frp.copy())
                fpLSThourlst.append(MapWrk.FP_hour.copy())
                fpLonlst.append(MapWrk.FP_lon.copy())
                fpLatlst.append(MapWrk.FP_lat.copy())
                nFires += MapWrk.FP_frp.shape[0]
                dayslst.append(copy.copy(today))
                if today.day == 1 and today.month == 1: print(today, len(fpFRPlst))
            today += spp.one_day
        fpFRP = np.array(fpFRPlst)
        fpLSThour = np.array(fpLSThourlst)
        fpLon = np.array(fpLonlst)
        fpLat = np.array(fpLatlst)
        days = np.array(dayslst)
        #
        # All fires collected. Store them for faster processing next time.
        with open(os.path.join(chOutDir,'all_fires.pickle'), 'wb') as handle:
            pickle.dump((fpFRP, fpLSThour, fpLon, fpLat, days, nFires), 
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Initiate metadata
        MapWrk = daily_maps(log=log)
        MapWrk.from_file(tStart.strftime(chInTemplate))
        # Read the stored fires
        with open(os.path.join(chOutDir,'all_fires.pickle'), 'rb') as handle:
            fpFRP, fpLSThour, fpLon, fpLat, days, nFires = pickle.load(handle)

    if ifAnalyseData:
        #
        # Analysis:
        # For each day, hours 1-3 are for the night overpass, 11-13 are for the day
        # Land use coordinates have to be taken from the right place
        #
        P_night = np.zeros(shape=(len(land_use.LUtypes)))
        P2_night = np.zeros(shape=(len(land_use.LUtypes)))
        N_night = np.zeros(shape=(len(land_use.LUtypes)))
        P_day = np.zeros(shape=(len(land_use.LUtypes)))
        P2_day = np.zeros(shape=(len(land_use.LUtypes)))
        N_day = np.zeros(shape=(len(land_use.LUtypes)))
        nFiresLU = np.zeros(shape=(len(land_use.LUtypes)))
        for it, today in enumerate(days):
            if today.day == 1 and today.month == 1: print(today)
            LUs = land_use.get_LU_4_fires(fpLon[it], fpLat[it])
            for iLU, LU in enumerate(land_use.LUtypes):
                idxLU = LUs == iLU
                if np.sum(idxLU) == 0: continue
                # Nighttime slice
                idxNight = np.isin(fpLSThour[it][idxLU], MapWrk.obsNightHrs)
                nHrsNightPresent = np.sum(np.isin(MapWrk.obsNightHrs, fpLSThour[it][idxLU]))
                FRPnight = fpFRP[it][idxLU][idxNight] 
#                FRPnight = fpFRP[it][idxLU][np.logical_and(fpLSThour[it][idxLU] >= 0, 
#                                                           fpLSThour[it][idxLU] <= 3)]
                N_night[iLU] += np.sum(idxNight)
                idxDay = np.isin(fpLSThour[it][idxLU], MapWrk.obsDayHrs)
                nHrsDayPresent = np.sum(np.isin(MapWrk.obsDayHrs, fpLSThour[it][idxLU]))
                FRPday = fpFRP[it][idxLU][idxDay] 
#                FRPday = fpFRP[it][idxLU][np.logical_and(fpLSThour[it][idxLU] >= 11, 
#                                                         fpLSThour[it][idxLU] <= 13)]
                N_day[iLU] += np.sum(idxDay)
                nFiresLU[iLU] += np.sum(idxLU)
                if N_night[iLU] * N_day[iLU] > 0:
                    if nHrsNightPresent > 0: 
                        P_night[iLU] += np.sum(FRPnight) / nHrsNightPresent
                        P2_night[iLU] += np.sum(np.square(fpFRP[it][idxLU][idxNight])) / nHrsNightPresent
#                    P2_night[iLU] += np.sum(np.square(fpFRP[it][idxLU]
#                                                      [np.logical_and(fpLSThour[it][idxLU] >= 0, 
#                                                                      fpLSThour[it][idxLU] <= 3)])) 
                    if nHrsDayPresent > 0:
                        P_day[iLU] += np.sum(FRPday)  / nHrsDayPresent
                        P2_day[iLU] += np.sum(np.square(fpFRP[it][idxLU][idxDay])) / nHrsDayPresent
#                    P2_day[iLU] += np.sum(np.square(fpFRP[it][idxLU]
#                                                    [np.logical_and(fpLSThour[it][idxLU] >= 11, 
#                                                                   fpLSThour[it][idxLU] <= 13)])) 
        print('N_day', N_day)
        print('N_night ', N_night)
        r_MODIS = P_day / (P_night + 1e-10)
        r_SEVIRI = np.sum(land_use_loc.diurnal[11:14,:], 
                              axis=0) / np.sum(land_use_loc.diurnal[0:4,:],axis=0)
        r_MODIS_per_fire = np.where(N_night == 0, 1.0, 
                                    (P_day / (N_day + 1e-10)) / (P_night / (N_night + 1e-10)+1e-10))
        r_SEVIRI_per_fire = np.sum(land_use_loc.diurnal_per_fire[11:13,:], 
                                   axis=0) / np.sum(land_use_loc.diurnal_per_fire[1:3,:],axis=0)
        print('LU   SEVIRI day/night  MODIS day/night nFires:')
        for iLU, LU in enumerate(land_use.LUtypes):
            print('%s %5.2f  %5.2f %i' % (LU, r_SEVIRI[iLU], r_MODIS[iLU], nFiresLU[iLU]))
    
        #
        # Recompute the diurnal profile using that of Seviri
        # Store the new variations and draw them
        #
        fOut = open(os.path.join(chOutDir,'diurnal_variations_new.ini'),'w')
        for diurnal, ratios, chTit in [(land_use_loc.diurnal, r_MODIS, '_total'), 
                                     (land_use_loc.diurnal_per_fire, r_MODIS_per_fire,
                                      '_per_fire')]:
            for iLU, LU in enumerate(land_use_loc.LUtypes):
                V_S = diurnal[:, iLU]
                R_dn = ratios[iLU]
                if nFiresLU[iLU] < 1000 or np.all(V_S == 1):
                    V_M = V_S
                    a = 1
                    b = 0
                else:
                    V_Sday = np.mean(V_S[11:14])
                    V_Snight = np.mean(V_S[1:4])
                    print(LU + ' day_night ratio%s SEVIRI= ' % chTit, V_Sday / V_Snight, 'MODIS= ', 
                          R_dn, np.sum(V_S)) 
                    #
                    # New V_M = a * V_S + b
                    a = (R_dn - 1.) / (V_Sday - 1. - (R_dn * (V_Snight - 1.)))
                    if a < 0: a = 1.0
                    b = (1. - a)
                    V_M = a * V_S + b
                    while np.any(V_M < 0.01):
                        fNeg = abs(np.sum(V_M[V_M<0.01] - 0.01))
                        print(LU + ' Handling negatives', LU, fNeg)
                        V_M = np.where(V_M < 0.01, 0.01 - fNeg / 24., V_M - fNeg / 24.)
                print(LU + ' New MODIS%s day-night ratio:' % chTit, 
                      np.mean(V_M[11:14]) / np.mean(V_M[1:4]), np.sum(V_M), np.sum(V_S))
                #
                # Draw diurnal profile for each LU
                fig, ax = mpl.pyplot.subplots()
                ax.plot(range(24), V_S, label='SEVIRI%s %s' % (chTit, LU))
                ax.plot(range(24), V_M, label='MODIS%s %s' % (chTit, LU))
                ax.legend()
                ax.set_title('%s %s a = %g, b = %g' % (LU, chTit, a,b))
                mpl.pyplot.savefig(os.path.join(chOutDir, 'pics', 'diurnal_' + LU + chTit + '.png'))
                mpl.pyplot.close()
                #
                # Store the new variations in the form suitable for the 
                #
                fOut.write('hour_in_day_index%s = %s %s\n' % (chTit, LU, ' '.join('%7.4f' % v for v in V_M)))
        fOut.close()
        #
        # Store the ratios
        with open(os.path.join(chOutDir,'ratios.pickle'), 'wb') as handle:
            pickle.dump((r_SEVIRI, r_MODIS, r_SEVIRI_per_fire, r_MODIS_per_fire, 
                         N_day, N_night, nFiresLU), 
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Read the previously written files
        with open(os.path.join(chOutDir,'all_fires.pickle'), 'rb') as handle:
            fpFRP, fpLSThour, fpLon, fpLat, days, nFires = pickle.load(handle)
        with open(os.path.join(chOutDir,'ratios.pickle'), 'rb') as handle:
            r_SEVIRI, r_MODIS, r_SEVIRI_per_fire, r_MODIS_per_fire, N_day, N_night, nFiresLU = pickle.load(handle)  
        print('Total number of fires: ', nFires)
    #
    # Make an overall barplot
    #
    #
    # Plot separate charts for _total and per_fire ratios
    for r_S, r_M, chTit in [(r_SEVIRI, r_MODIS, '_total'), 
                            (r_SEVIRI_per_fire, r_MODIS_per_fire, '_per_fire')]:
        fig, ax1 = mpl.pyplot.subplots(figsize = (20,10))
        ax2 = ax1.twinx()

        # Number of bars per group
        n_bars = 3   #len(land_use.LUtypes)
        # The width of a single bar
        bar_width = 0.5 / n_bars
        # List containing handles for the drawn bars, used for the legend
        bars = []
        names = []
        # Iterate over all data
        for i, (name, values, ax) in enumerate([('SEVIRI' + chTit, r_S, ax1), 
                                                ('MODIS' + chTit, r_M,  ax1),
                                                ('nbr of cases', nFiresLU, ax2)]):
            if i < 2:
                edgecolor = ''
                linewidth = 0
                color = mpl.pyplot.rcParams['axes.prop_cycle'].by_key()['color'][i]
            else:
                edgecolor = mpl.pyplot.rcParams['axes.prop_cycle'].by_key()['color'][i]
                linewidth = 1
                color = (0,0,0,0)
            names.append(name)
            # The offset in x direction of that bar
            x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
    
            if False: #data_stdev:
                # Draw a bar for every value of that type
                for x, y in enumerate(values):
                    bar = ax.bar(x + x_offset, y, width=bar_width, color=color, edgecolor=edgecolor,
    #                             yerr=data_stdev[name][x],
                                 linewidth = linewidth, error_kw={'elinewidth':0.5})
            else:
                # Draw a bar for every value of that type
                for x, y in enumerate(values):
                    bar = ax.bar(x + x_offset, y, width=bar_width, color=color, edgecolor=edgecolor,
                                 linewidth = linewidth)
    
            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])
    
        # Draw legend if we need
        ax1.legend(bars, names, fontsize=12)
        ax1.set_ylabel('day(hrs11:13LST) / night(hrs1:3LST) ratio',fontsize=12)
        mpl.pyplot.yticks(fontsize=12)
        ax1.grid(axis='y', color='lightgrey')
        ax2.set_ylabel('number of cases',fontsize=12)
        mpl.pyplot.yticks(fontsize=12)
        mpl.pyplot.yscale('symlog', ax=ax2, linthreshy=1000)
    
        mpl.pyplot.xticks(range(len(r_SEVIRI)), land_use_loc.LUtypes)
        mpl.pyplot.setp( ax1.xaxis.get_majorticklabels(), rotation=86, fontsize=12)
    
        mpl.pyplot.savefig(os.path.join(chOutDir, 'ratio_bars%s.png' % chTit), dpi=200, bbox_inches='tight')
        mpl.pyplot.close()


#=======================================================================

def verify_diurnal_var(land_use, chOutDir):
    #
    # Just checks the new formulas for the diurnal variation
    #
    i_night = 1  # local hour of the first observations NIGHT
    i_day = 11 # local hour of the second observation DAY
    FRPnight = 4   # FRP night MW
    FRPday = 20   # FRP daytime MW
    LUref = 'EU_grass'   # for instance
    iLU = np.searchsorted(land_use.LUtypes, LUref)
    R_dn = FRPday / FRPnight
    V_S = land_use.diurnal[:, iLU]
    V_Sday = V_S[i_day]
    V_Snight = V_S[i_night]
    print('day_night ration SEVIRI= %g, MODIS= %g' % (V_Sday / V_Snight, R_dn), np.sum(V_S)) 
    #
    # New V_M = a * V_S + b
    a = (R_dn - 1.) / (V_Sday - 1. - (R_dn * (V_Snight - 1.)))
    b = (1. - a)
    V_M = a * V_S + b
    print(zip(V_S, V_M, V_S * a + b))
    print('New MODIS day-night ratio:', V_M[i_day] / V_M[i_night], np.sum(V_M), np.sum(a * V_S + b), np.sum(V_S))

    fig, ax = mpl.pyplot.subplots()
    ax.plot(range(24), land_use.diurnal[:, iLU], label='SEVIRI ' + LUref)
    ax.plot(range(24), V_M, label='MODIS ' + LUref)
    ax.legend()
    ax.set_title('a = %g, b = %g' % (a,b))
    mpl.pyplot.savefig(os.path.join(chOutDir, 'diurnal_' + LUref + '.png'))
    mpl.pyplot.close()


#=======================================================================

def get_LU_contrib_2_FRP(land_use, fire_src_templ, tStart, tEnd, QA, wrkDir, chFNmOut, mpisize, mpirank, communicator):
    #
    # Scans all available years and gets the amount of energy released by each LU burns
    # Can be used as weighting coefficients for diurnal variaiton etc.
    #
    MapWrk = daily_maps(log = spp.log(os.path.join(wrkDir,'run.log')))
    MapWrk.from_file(tStart.strftime(fire_src_templ))
    LU_cnt = np.zeros(shape=(len(land_use.LUtypes),MapWrk.grid.ny,MapWrk.grid.nx),dtype=np.int32)
    LU_frp = np.zeros(shape=(len(land_use.LUtypes),MapWrk.grid.ny,MapWrk.grid.nx),dtype=np.float32)
    #
    # The main cycle over time, done in parallel by all processes
    #
    today = tStart
    iProcess = 0
    nFiresTot = 0
    nFiresBad = 0
    while today <= tEnd:
        if np.mod(iProcess, mpisize) != mpirank:
            iProcess += 1
            today += spp.one_day
            continue
        if today.day == 1:
            print(today, "mpisize, mpirank:", mpisize, mpirank, '; Fires total, removed: ', nFiresTot, nFiresBad)

        MapWrk.from_file(today.strftime(fire_src_templ))
        if MapWrk.FP.nFires > 0:
            #
            # Since this sub is within LST, no BAD_DAY checks allowed: the BAD_DAY is a definitin in UTC
            # So, only FULL QA would make any effect
            #
            if QA.QA_action == 'FULL':
#             print('today, before cleaning FRP and nFires:', today, np.sum(MapWrk.FP.FRP), MapWrk.FP.FRP.size)
                iBad = QA.get_mask(MapWrk.FP.lon, MapWrk.FP.lat, MapWrk.FP.FRP,
                                   MapWrk.FP.LU, today, MapWrk.QA)
                nFiresTot += MapWrk.FP.nFires
                nFiresBad += np.sum(iBad)
                MapWrk.remove_bad_fires(np.argwhere(iBad))
                
            FP_LU = land_use.get_LU_4_fires(MapWrk.FP.lon, MapWrk.FP.lat)
            # Add LU fires: count the occasions and sum-up FRP
            # Note that we cannot promise that the LU in the fire file is the same as ours
            try:
                LU_cnt[FP_LU,MapWrk.FP.iy, MapWrk.FP.ix] += 1
                LU_frp[FP_LU,MapWrk.FP.iy, MapWrk.FP.ix] += MapWrk.FP.frp
            except:
#                print(FP_LU,MapWrk.FP.iy, MapWrk.FP.ix)
                FP_LU = land_use.get_LU_4_fires(MapWrk.FP.lon, MapWrk.FP.lat)
                LU_cnt[FP_LU,MapWrk.FP.iy, MapWrk.FP.ix] += 1
                LU_frp[FP_LU,MapWrk.FP.iy, MapWrk.FP.ix] += MapWrk.FP.FRP
        iProcess += 1
        today += spp.one_day
    #
    # Store the intermediate sums
    spp.ensure_directory_MPI(os.path.join(wrkDir,'tmp'))
    with open(os.path.join(wrkDir,'tmp','LUcontrib_tmp_%03i.pickle' % mpirank), 'wb') as handle:
        pickle.dump((LU_cnt, LU_frp), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # wait for all
    spp.MPI_join('fires_vs_landuse_time_cycle', wrkDir, mpisize_loc, mpirank_loc, communicator, spp.one_hour * 2.0)
    
    # non-zero MPI processes are free to go
    if mpirank != 0: return
    #
    # MPI 0 reads them all and adds up to its own vars
    for m in range(1, mpisize_loc):
        with open(os.path.join(wrkDir,'tmp','LUcontrib_tmp_%03i.pickle' % m), 'rb') as handle:
            LU_cnt_tmp, LU_frp_tmp = pickle.load(handle)
        LU_cnt += LU_cnt_tmp
        LU_frp += LU_frp_tmp

    #
    # Compute the total FRP per grid cell and fractions of frp from each land use
    # How many LUs contributed - max?
    mapFRP_tot = np.sum(LU_frp,axis=0)
    mapNFires_tot = np.sum(LU_cnt, axis=0)
    nContributors = np.max(np.sum(LU_cnt>0,axis=0))
    argsortFRP = np.argsort(LU_frp,axis=0)[::-1,:,:]
    argsortNFires = np.argsort(LU_cnt, axis=0)[::-1,:,:]
    LU_fract_frp = np.take_along_axis(LU_frp, argsortFRP, 0)/ (mapFRP_tot[None,:,:] + 1e-10)
    LU_fract_nFires = np.take_along_axis(LU_cnt, argsortNFires, 0)/ (mapNFires_tot[None,:,:] + 1e-10)

    # Temporary
#    diurnal_map = np.take_along_axis(land_use.diurnal[:,:,None,None], argsortFRP[None,:,:,:], 1)
#    LU_weighted_diurnal_FRP = np.sum(LU_fract_frp[None,:,:,:] * diurnal_map[:,:,:,:],axis=1)
    #
    # The array below can be huge 
    #
    try:
        LU_weighted_diurnal_FRP = np.sum(LU_fract_frp[None,:,:,:] * 
                                         np.take_along_axis(land_use.diurnal[:,:,None,None], 
                                                            argsortFRP[None,:,:,:], 1), axis=1)
    except:
        LU_weighted_diurnal_FRP = np.zeros(shape=(land_use.diurnal.shape[0],   # 24 hrs
                                                  LU_fract_frp.shape[1],LU_fract_frp.shape[2]))  # grid
#        print(LU_weighted_diurnal_FRP.shape, argsortFRP.shape, land_use.diurnal.shape)

        for iHr in range(land_use.diurnal.shape[0]):
            LU_weighted_diurnal_FRP[iHr,:,:] = np.sum(LU_fract_frp[:,:,:] * 
                                                      np.take_along_axis(land_use.diurnal[iHr,:,None,None], 
                                                                         argsortFRP[:,:,:], 0), axis=0)
    #
    # Now, store the obtained fields to the nc file
    # Tricky file, a few unusual dimensions and variables 
    outF = silamfile.open_ncF_out(chFNmOut,    #os.path.join(outDir,'LU_FRP_contrib_%s.nc4' % MapWrk.gridName),
                                  'NETCDF4', MapWrk.grid, silamfile.SilamSurfaceVertical(), 
                                  tStart, [tStart], [], 
                                  ['FRP_raw', 'nFires_tot',], 
                                  {'FRP_raw':'MW', 'nFires_tot':''},
                                  -999., True, 3, None)
    # Store the variables
    outF['FRP_raw'][0,:,:] = mapFRP_tot[:,:]
    outF['nFires_tot'][0,:,:] = mapNFires_tot[:,:]
    # Contributions and fractions (nContributors, ny, nx)
    # Dimension
    outF.createDimension("LU_contrib", nContributors)
    outF.createDimension("hours_of_day", 24)
    # variables
    for varNm, varType, varVals in [('contributors_FRP', 'i2', argsortFRP), 
                                    ('contributors_nFires', 'i2', argsortNFires), 
                                    ('fraction_FRP', 'f4', LU_fract_frp),
                                    ('fraction_nFires', 'f4', LU_fract_nFires)]:
        outF.createVariable(varNm, varType, ('LU_contrib', 'lat', 'lon'), zlib=True, complevel=5, 
                            chunksizes=(1,MapWrk.grid.ny,MapWrk.grid.nx), fill_value=-999)
        
        outF.variables[varNm][:,:,:] = varVals[:nContributors,:,:]

    # Store the map of mean weighted diurnal variations
    outF.createVariable('LU_weighted_diurnal_coef', 'f4', ('hours_of_day', 'lat', 'lon'), zlib=True, 
                        complevel=5, chunksizes=(1, MapWrk.grid.ny, MapWrk.grid.nx), fill_value=-999)
    outF.variables['LU_weighted_diurnal_coef'][:,:,:] = LU_weighted_diurnal_FRP[:,:,:]

    # And land use names and axes    
    land_use.to_nc_file(outF)

#    for i, c in enumerate(outF.variables['land_use_code'][:]):
#        chLU = nc4.chartostring(c[:]).item()
#        MapWrk.log.log('LU: %g, %s, %g fires, %g MWday' % (i, chLU, nFires_tot[chLU], FRP_tot[chLU]))

    for i, c in enumerate(outF.variables['land_use_code'][:]): 
        print(i, nc4.chartostring(c[:]))




#=======================================================================

def FRP_2_map_LU_split(land_use, gridOut, fire_src_templ, tStart, tEnd, tStep, outDir, chOutFNm, 
                       mpisize, mpirank, communicator):
    #
    # Scans all available years and gets the amount of energy released by each LU burns
    # Can be used as weighting coefficients for diurnal variaiton etc.
    #
    MapWrk = daily_maps(log = spp.log(os.path.join(outDir,'run.log')))
    MapWrk.from_file(tStart.strftime(fire_src_templ))
    LU_cnt = np.zeros(shape=(len(land_use.LUtypes),gridOut.ny,gridOut.nx),dtype=np.int16)
    LU_frp = np.zeros(shape=(len(land_use.LUtypes),gridOut.ny,gridOut.nx),dtype=np.float32)
    #
    # The main cycle over time, done in parallel by all processes
    #
    today = tStart
    iProcess = 0
    while today <= tEnd:
        if np.mod(iProcess, mpisize) != mpirank:
            iProcess += 1
            today += tStep  #spp.one_day
            continue
        if today.day == 1:
            print(today, "mpisize, mpirank:", mpisize, mpirank)
        MapWrk.from_file(today.strftime(fire_src_templ))
        if MapWrk.FP.nFires > 0:
            # get LUs and project fire points to the output grid 
            FP_LU = land_use.get_LU_4_fires(MapWrk.FP_lon, MapWrk.FP_lat)
            fx, fy = gridOut.geo_to_grid(MapWrk.FP_lon, MapWrk.FP_lat)
            ix = np.mod(np.round(fx).astype(np.int32),gridOut.nx) 
            iy = np.round(fy).astype(np.int32)
            # Add LU fires: count the occasions and sum-up FRP
            # Note that we cannot promise that the LU in the fire file is the same as ours
            if np.any(MapWrk.FP_frp > 500):
                print('>>> Strange FRP:', today, MapWrk.FP_frp[MapWrk.FP_frp>500])
                MapWrk.FP_frp[MapWrk.FP_frp > 1000] = 0
                MapWrk.FP_frp[MapWrk.FP_frp > 500] = 500
            
            LU_cnt[FP_LU, iy, ix] += 1
            LU_frp[FP_LU, iy, ix] += MapWrk.FP_frp
        iProcess += 1
        today += tStep  #spp.one_day
    #
    # Store the intermediate sums
#    print('Starting pickle')
#    spp.ensure_directory_MPI(os.path.join(outDir,'tmp'))
#    with open(os.path.join(outDir,'tmp','FRP_LU_wise_tmp_%03i.pickle' % mpirank), 'wb') as handle:
#        pickle.dump((LU_cnt, LU_frp), handle, protocol=pickle.HIGHEST_PROTOCOL)
#    # wait for all
#    spp.MPI_join('FRP_sum_LU_wise', outDir, spp.one_hour * 2.0)
#    #
#    # non-zero MPI processes are free to go, they done their job
#    #
#    if mpirank != 0: return
#    #
#    # MPI 0 reads them all and sums up
#    print('Starting pickle reading')
#    for m in range(1, mpisize_loc):
#        with open(os.path.join(outDir,'tmp','FRP_LU_wise_tmp_%03i.pickle' % m), 'rb') as handle:
#            LU_cnt_tmp, LU_frp_tmp = pickle.load(handle)
#        LU_cnt += LU_cnt_tmp
#        LU_frp += LU_frp_tmp
    #
    # Compute the total FRP per grid cell and fractions of frp from each land use
    # How many LUs contributed - max?
    mapFRP_tot = np.sum(LU_frp,axis=0)
    FRP_tot = {'LU_all':np.sum(mapFRP_tot)}
    FRP_99_9 = np.percentile(mapFRP_tot,99)
    nZeroFRPCells = np.sum(mapFRP_tot==0)
    nPositFRPCells = np.sum(mapFRP_tot>0)
    mapNFires_tot = np.sum(LU_cnt, axis=0)
    nFires_tot = {'LU_all':np.sum(mapNFires_tot)}
    nFires_99_9 = np.percentile(mapNFires_tot,99)
    nZeroNFiresCells = np.sum(mapNFires_tot==0)
    nPositNFiresCells = np.sum(mapNFires_tot>0)

    histFRP, bin_edges_FRP = np.histogram(mapFRP_tot[mapFRP_tot>0], bins=1000)
    histNFires, bin_edges_nFires = np.histogram(mapNFires_tot[mapNFires_tot>0],
                                                bins=min(1000, np.max(mapNFires_tot)))
    MapWrk.log.log('FRP_2_map_LU_split\n timeStart %s\ntimeEnd %s' % 
                   (tStart.strftime('%Y%m%d %H%M'), tEnd.strftime('%Y%m%d %H%M')))
    MapWrk.log.log('Bin edges FRP:\n' + ' '.join(('%g' %v for v in bin_edges_FRP)))
    MapWrk.log.log('Histogram FRP:\n' + ' '.join(('%g' %v for v in histFRP)))
    MapWrk.log.log('\nbin edges nFires\n' + ' '.join(('%g' %v for v in bin_edges_nFires)))
    MapWrk.log.log('Histogram nFires\n' + ' '.join(('%g' %v for v in histNFires)))
    MapWrk.log.log('\nTotal FRP %g , nFires %g\n' % (FRP_tot['LU_all'], nFires_tot['LU_all']))
    
    idxLrgFRP = np.argwhere(mapFRP_tot > np.max(mapFRP_tot) * 0.5)
    idxManyFires = np.argwhere(mapNFires_tot > np.max(mapNFires_tot) * 0.5)
    lonLrgFRP, latLrgFRP = gridOut.grid_to_geo(idxLrgFRP[:,1], idxLrgFRP[:,0])
    lonManyFires, latManyFires = gridOut.grid_to_geo(idxManyFires[:,1], idxManyFires[:,0])
    for i in range(lonLrgFRP.size):
        MapWrk.log.log('Large FRP (ix, iy, lon, lat, FRP, nFires): [%g %g] (%g %g) %g, %g' % 
                       (idxLrgFRP[i,1], idxLrgFRP[i,0], lonLrgFRP[i], latLrgFRP[i], 
                        mapFRP_tot[idxLrgFRP[i,0],idxLrgFRP[i,1]], 
                        mapNFires_tot[idxLrgFRP[i,0],idxLrgFRP[i,1]]))
    MapWrk.log.log('\n')
    for i in range(lonManyFires.size):
        MapWrk.log.log('Many Fires (ix, iy, lon, lat, FRP, nFires): [%g %g] (%g %g) %g, %g' % 
                       (idxManyFires[i,1], idxManyFires[i,0], lonManyFires[i], latManyFires[i], 
                        mapFRP_tot[idxManyFires[i,0],idxManyFires[i,1]],
                        mapNFires_tot[idxManyFires[i,0],idxManyFires[i,1]]))
    MapWrk.log.log('\n99-th percentile for FRP and nFires: %g %g' % (np.percentile(mapFRP_tot,99),
                                                                     np.percentile(mapNFires_tot,99)))
    MapWrk.log.log('99.9-th percentile for FRP and nFires: %g %g\n' % (FRP_99_9, nFires_99_9))
    #
    # Draw the histograms with some basic numbers
    #
    for iLU, chLU in enumerate(np.concatenate((['LU_all'], land_use.LUtypes), axis=0)):
        fig, axes = mpl.pyplot.subplots(2,1, figsize=(6,6))
        axFRP = axes[0]
        axNFires = axes[1]
        if chLU != 'LU_all': 
            histFRP, bin_edges_FRP = np.histogram(LU_frp[iLU-1,:,:][LU_frp[iLU-1,:,:]>0], bins=1000)
            FRP_tot[chLU] = np.sum(LU_frp[iLU-1,:,:])
            histNFires, bin_edges_nFires = np.histogram(LU_cnt[iLU-1,:,:][LU_cnt[iLU-1,:,:]>0], 
                                                        bins=min(1000,np.max(LU_cnt[iLU-1,:,:])+1))
            nFires_tot[chLU] = np.sum(LU_cnt[iLU-1,:,:])
            nZeroFRPCells = np.sum(LU_frp[iLU-1,:,:]==0)
            nPositFRPCells = np.sum(LU_frp[iLU-1,:,:]>0)
            nZeroNFiresCells = np.sum(LU_cnt[iLU-1,:,:]==0)
            nPositNFiresCells = np.sum(LU_cnt[iLU-1,:,:]>0)
            if np.sum(LU_frp[iLU-1,:,:]) == 0:
                FRP_99_9 = 0
                nFires_99_9 = 0
            else:
                FRP_99_9 = np.percentile(LU_frp[iLU-1,:,:][LU_frp[iLU-1,:,:]>0], 99.9)
                nFires_99_9 = np.percentile(LU_cnt[iLU-1,:,:][LU_cnt[iLU-1,:,:]>0], 99.9)
        axes[0].semilogy(bin_edges_FRP[1:], histFRP, label='FRP')
        axes[0].set_xlabel('FRP, MWdays')
        axes[0].set_ylabel('Nbr of cases, FRP>0')
        axes[0].set_title('FRP histogram, ' + chLU)
        xMin, xMax = axes[0].get_xlim()
        yMin, yMax = axes[0].get_ylim()
        axes[0].text((xMin + xMax)*0.5, np.sqrt((yMin+1)*(yMax+1)),  # bin_edges_FRP[-1]*0.5, np.max(histFRP)*0.01,
                     'FRP_tot %g MWday\n99.9%s = %g MWday\nNbr of zero cells %g\nNbr of >0 cells = %g' % 
                     (FRP_tot[chLU], "%", FRP_99_9, nZeroFRPCells, nPositFRPCells))
        axes[1].semilogy(bin_edges_nFires[1:], histNFires, label='nFires')
        axes[1].set_xlabel('Nbr of fires')
        axes[1].set_ylabel('Nbr of cases nFire>0')
        axes[1].set_title('Nbr of fires histogram, ' + chLU)
        xMin, xMax = axes[1].get_xlim()
        yMin, yMax = axes[1].get_ylim()
        axes[1].text((xMin + xMax)*0.5, np.sqrt((yMin+1)*(yMax+1)),   #bin_edges_nFires[-1]*0.5, np.max(histNFires)*0.01,
                     'nFires_tot %g\n99.9%s = %g\nNbr of zero cells = %g\nNbr of >0 cells = %g' % 
                     (nFires_tot[chLU], "%",nFires_99_9, nZeroNFiresCells, nPositNFiresCells))
        mpl.pyplot.tight_layout()
        mpl.pyplot.savefig(os.path.join(outDir,'histogr_%s.png' % chLU), dpi=200)
        mpl.pyplot.clf()
        mpl.pyplot.close()
        
    #
    # Now, store the obtained fields to the nc file
    # Tricky file, a few unusual dimensions and variables
    # 
    outF = silamfile.open_ncF_out(os.path.join(outDir,chOutFNm),
                                  'NETCDF4', gridOut, silamfile.SilamSurfaceVertical(), 
                                  tStart, [tStart], [], 
                                  ['FRP_raw', 'nFires_tot',], 
                                  {'FRP_raw':'MW', 'nFires_tot':''},
                                  -999., True, 3, None)
    # Store the variables
    outF['FRP_raw'][0,:,:] = mapFRP_tot[:,:]
    outF['nFires_tot'][0,:,:] = mapNFires_tot[:,:]
    # And land use names and axes    
    land_use.to_nc_file(outF)

    # Contributions and fractions (nContributors, ny, nx)
    # variables
    for varNm, varType, varVals in [('FRP_4_LU', 'i2', LU_frp),
                                    ('nFires_4_LU', 'i2', LU_cnt)]:
        print(varNm)
        outF.createVariable(varNm, varType, ('LU_types', 'lat', 'lon'), zlib=True, complevel=5, 
                            chunksizes=(1,MapWrk.grid.ny,MapWrk.grid.nx), fill_value=-999)
        
        outF.variables[varNm][:,:,:] = varVals[:,:,:]
        print('Done')

    for i, c in enumerate(outF.variables['land_use_code'][:]):
        chLU = nc4.chartostring(c[:]).item()
        MapWrk.log.log('LU: %g, %s, %g fires, %g MWday' % (i, chLU, nFires_tot[chLU], FRP_tot[chLU]))

    outF.close()

    

#############################################################################
#############################################################################

if __name__ == '__main__':
    print('Hi')

    ifProcess_IS4FIRES_v3_0 = False
    ifConvert2LocalTime = False
    ifProcess_IS4FIRES_v2_0 = False
    ifCheckFile = False
    ifCheckDuplicates = False
    ifParameteriseDiurnalVar = False
    ifFRP_tsM_to_maps = False
    ifFRP_fire_dots_to_maps = True

    grids = [#(gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_EU_2_0.txt'),'EU_2_0'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_EU_0_25.txt'),'EU_0_25'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_EU_0_5.txt'),'EU_0_5'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_EU_1_0.txt'),'EU_1_0'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_EU_1_5.txt'),'EU_1_5'),
             (gridtools.fromCDOgrid('d:\\data\\emis\\fires\\grd_glob_3_0.txt'),'glob_3_0'),
             (gridtools.fromCDOgrid('d:\\data\\emis\\fires\\grd_glob_0_5.txt'),'glob_0_5'),
#             (gridtools.fromCDOgrid('d:\\data\\emis\\fires\\grd_glob_0_1.txt'),'glob_0_1'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_glob_1_0.txt'),'glob_1_0'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_glob_1_5.txt'),'glob_1_5'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_glob_2_0.txt'),'glob_2_0'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_glob_0_5.txt'),'glob_0_5'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\grd_glob_0_01.txt'),'glob_0_01'),
#             (gridtools.fromCDOgrid('d:\\project\\Fires\\forecasting_v2_0\\GRIP4_grid_global.txt'),'GRIP4_grd_glob'),
            ]
#    chDirEcodata = '/projappl/project_2004363/fires'
#    chDirWrk = '/scratch/project_2004363/fires/IS4FIRES_v3_0_VIIRS'
#    chDirEcodata = 'd:\\data\\emis\\fires'
    chDirEcodata = 'd:\\project\\Fires'
    chDirWrk = 'e:\\project\\fires\\'
    chFireMetadataFNm = path.join(chDirEcodata,
#                                  'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_clean_BC_OC_filled.txt')
#                                  'fire_metadata_ecodata_Acagi_PM_v5_5_cb4_IS4FIRES_v3_0_noPeat_nshrub_CB5_modified_halo_BC_OC.ini')
                                  'fire_metadata_ecodata_cbm7_continents_v2_optim_MASfit.ini')
    
    dayStart = dt.datetime(2018,6,2)
#    dayEnd = dt.datetime(2020,8,13)
    dayEnd = dt.datetime(2019,9,24)

    chInTemplate = path.join(chDirWrk, '$GRD_LST', 'IS4FIRES_v3_0_$GRD_%Y%m%d_LST.nc4')
    
    #
    # Creating the IS4FIRES v.3.0 intermediate dataset
    #
    if ifProcess_IS4FIRES_v3_0:
        chDirMain = 'f:\\data\\MODIS_raw_data\\'
        chDirWrk = 'd:\\project\\fires\\forecasting\\IS4FIRES_v3_0_gridded\\tst'
        cloud_threshold = 0.6
        if not path.exists(chDirWrk): os.makedirs(chDirWrk)
        MxD14_templ = path.join(chDirMain,'MxD14_coll_6_extract', '%Y', '%Y.%m.%d', 
                                   'MxD14.A%Y%j.%H%M.006.*.hdf.nc4')
        MxD35_templ = path.join(chDirMain,'MxD35_L2_extract', '%Y', '%j',
                                   'MxD35_L2.A%Y%j.%H%M.061.*hdf_extract.nc4')
        iProcess = 0
        for grid_def in grids:
            time.sleep(mpirank_loc)
            # Local-solar-time maps are arranged as a dictionary with day as a key
            today = dt.datetime(2003,1,1)
            log = spp.log(path.join(chDirWrk,'run_IS4FIRES_v3_0_%s_%02i_tst.log' % (grid_def[1], 
                                                                                    mpirank_loc)))
            while today < dt.datetime(2003,12,31):
                if np.mod(iProcess, mpisize_loc) != mpirank_loc: 
                    iProcess += 1
                    today += spp.one_day
                    continue
                print(today, 'MPI rank:', mpirank_loc)
                nFires = 0
                # Create a map for today
                Map4day = daily_maps(grid_def, today, cloud_threshold, False, 
                                     24, LU_module.land_use(chFireMetadataFNm), None, log)
                # collect the data
                Map4day.Fill_daily_maps(MxD14_templ, MxD35_templ, ['MOD','MYD'],
                                        spp.one_minute * 5, True)
                # store the files
                if not path.exists(path.join(chDirWrk, grid_def[1])): 
                    os.makedirs(path.join(chDirWrk, grid_def[1]))
                Map4day.to_files(path.join(chDirWrk, grid_def[1],
                                              'IS4FIRES_v3_0_%GRD_%Y%m%d.nc4'.replace('%GRD',
                                                                                      grid_def[1])))
                iProcess += 1
                today += spp.one_day
    #
    # Converting UTC daily maps to local-time daily maps
    #
    if ifConvert2LocalTime:
        chMainDir = 'f:\\project\\fires\\IS4FIRES_v3_0_gridded'
        iProcess = 0
        for grid_def in grids:
            if np.mod(iProcess, mpisize_loc) != mpirank_loc: 
                iProcess += 1
                continue
            # Local-solar-time maps are arranged as a dictionary with day as a key
            log = spp.log(path.join(chMainDir, 'run_IS4FIRES_v3_0_%s_%02i.log' % (grid_def[1], 
                                                                                     mpirank_loc)))
            if not path.exists(path.join(chMainDir, grid_def[1] +'_LST')):
                os.makedirs(path.join(chMainDir, grid_def[1] +'_LST'))

            templInput = path.join(chMainDir, grid_def[1],'IS4FIRES_v3_0_%GRD_%Y%m%d.nc4'.
                                      replace('%GRD',grid_def[1])),
            templOutput = path.join(chMainDir, grid_def[1] + '_LST', 
                                       'IS4FIRES_v3_0_%GRD_%Y%m%d_LST.nc4'.replace('%GRD',grid_def[1]))

            MapWrk = daily_maps(log = log)
            dicOutMaps = {}

            today = dayStart
            while today <= dayEnd:
                #
                # Read the input data from a netCDF file.
                MapWrk.from_file(today.strftime(templInput))   # nMaps == 1 means only unified map in the file
                #
                # Convert time
                MapWrk.UTC_vs_local_time('to_local_time', dicOutMaps)
                #
                # yesterday map can now be stored
                dayOut = today - spp.one_day
                dicOutMaps[dayOut].to_files(dayOut.strftime(templOutput))  # ifUnifiedMapOnly
                #
                # ... and removed
                dicOutMaps.pop(dayOut)

                today += spp.one_day 
    #
    # Re-creating the IS4FIRES v.2.0 dataset
    #
    if ifProcess_IS4FIRES_v2_0:
        product_FRP = '14'
        suffix_FRP = '_L2'
        product_cld = '35'
        suffix_cld = '_nc4'
        chDirMain = 'f:\\data\\MODIS_raw_data\\tmp\\%s%s%s'
        chDirWrk = 'd:\\project\\fires\\forecasting'
    #    today = dt.datetime(2020,8,12)
        today = dt.datetime(2010,12,31)
        iProcess = 0
        while today > dt.datetime(2009,12,31):
    #    while today > dt.datetime(2000,2,1):
            if np.mod(iProcess, mpisize_loc) != mpirank_loc: 
                iProcess += 1
                today -= spp.one_day
                continue
            print(today, 'MPI rank:', mpirank_loc)
            arGran = []
            nFires = 0
            satAbbrev = 'MxD'
            for satellite in ['MYD']: #,'MOD']:
                now = today
                while now < today + spp.one_day:
                    arGran.append(sgMOD.granule_MODIS(satellite, now, 
                                                 path.join(chDirMain % (satellite, product_FRP, suffix_FRP), 
                                                              '%Y', '%j', #%Y.%m.%d',
                                                              satellite + product_FRP + 
                                                              '.A%Y%j.%H%M.006.*.hdf.nc4'),
                                                 path.join(chDirMain % (satellite, product_cld, suffix_cld), 
                                                              '%Y','%j',
                                                              satellite + product_cld + 
                                                              '_L2.A%Y%j.%H%M.061.*.hdf.nc4'), 
                                                 spp.log(path.join(chDirWrk,'test_MODIS.log'))))
                    if arGran[-1].pick_granule_data_IS4FIRES_v2_0():
                        if arGran[-1].FP_frp.shape[0] > 0:
                            nFires += arGran[-1].FP_frp.shape[0]
                            print('Number of fires:', arGran[-1].FP_frp.shape[0])
                        else:
                            arGran.pop(-1)
                            print('No fires')
                    else:
                        arGran[-1].log.log('Failed satellite data reading')
                        arGran.pop(-1)
                        print('Failed reading')
                    now += spp.one_minute * 5
            fOut = open(path.join(chDirWrk,'MODIS_2010_fs1','%s14_fs1' % satAbbrev,
                                  today.strftime('IS4FIRES_v2_0_%Y%m%d_'+satAbbrev+'.fs1')), 'w')
            fOut.write('FRP_DATASET_V1\n number_of_fires = %i\n' % nFires + 
                       ' max_number_of_same_fire_observations = 1\n')
            fOut.write('# fireNbr yr mon day hr min sec   lon     lat      dS      dT     km   frp  MW' + 
                       '    T4      T4b     T11    T11b    TA    MCE  FireArea SZA\n')
            iFireCnt = 1
            for granule in arGran:
                iFireCnt = granule.write_granule_IS4FIRES_v2_0(iFireCnt, fOut)
            fOut.write('END_FRP_DATASET_V1\n')
            fOut.close()
            iProcess += 1
            today -= spp.one_day
    
    
    if ifCheckFile:
        check_file('f:\\project\\fires\\IS4FIRES_v3_0_gridded\\glob_2_0_LST\\IS4FIRES_v3_0_glob_2_0_20121114_LST.nc4')


    if ifCheckDuplicates:
        chDirMODIS_raw = 'f:\\data\\MODIS_raw_data\\'
        chJunkDir = 'f:\\data\\MODIS_raw_data\\duplicates'
        find_duplicate_files(tStart = dt.datetime(2020,1,1),
                             tEnd = dt.datetime(2020,3,31),
                             tStep = spp.one_minute * 5,
                             MxD14_templ = path.join(chDirMODIS_raw,
                                                     'MxD14_coll_6_extract', '%Y', '%Y.%m.%d', 
                                                     'MxD14.A%Y%j.%H%M.006.*.hdf.nc4'), 
                             MxD35_templ = path.join(chDirMODIS_raw,
                                                     'MxD35_L2_extract', '%Y', '%j',
                                                     'MxD35_L2.A%Y%j.%H%M.061.*hdf_extract.nc4'),
                             chJunkDir = chJunkDir)


    if ifParameteriseDiurnalVar:
        chDirMetadata = 'd:\\project\\Fires\\forecasting_v2_0'   #/fmi/projappl/project_2001411/fires'
    
        # Ecosystem metadata file - either globally 7 LUs or continent-wise
        land_use_loc = LU_module.land_use(chFireMetadataFNm)
        chOutDir = 'd:\\project\\fires\\diurnal_variation\\IS4FIRES_v3_0'
        if not os.path.exists(chOutDir): os.makedirs(os.path.join(chOutDir,'pics'))

        # Get the diurnal variation from the land use and MODIS fire reports
        #
        get_day_night_ratio(False, True, land_use_loc, chInTemplate.replace('$GRD',grids[0][1]), 
                            dt.datetime(2000,3,1), dt.datetime(2020,8,11),
                            chOutDir, spp.log(os.path.join(chOutDir,'get_diurnal.log')))

    if ifFRP_tsM_to_maps:
        for chFNm in glob.glob('f:\\project\\fires\\IS4FIRES_v3_0_grid_FP\\EU_2_0_LST_daily\\*tsMatrix*nc4'):
            print(chFNm)
            MyTimeVars.TsMatrix.fromNC(chFNm).to_map(grids[0][0], -999, chFNm.replace('tsMatrix','map'))

        
    if ifFRP_fire_dots_to_maps:
        land_use_loc = LU_module.land_use(chFireMetadataFNm)
        stepDays = 16
#        grid_loc = silamfile.SilamNCFile(land_use_loc.chMapFNm).grid
        for grid_loc in grids:
#            for startDay in list((dt.datetime(2018,1,5) + 
#            for startDay in list((dt.datetime(2012,1,20) + 
            for startDay in list((dt.datetime(2003,1,1) + 
                                  spp.one_day * i for i in range(stepDays))):
                dirOut = os.path.join(chDirWrk,
#                                      'FRP_LU_wise_VIIRS_%s_start%s_step%id' % 
                                      'FRP_LU_wise_MODIS_%s_start%s_step%id' % 
                                      (grid_loc[1], startDay.strftime('%Y%m%d'), stepDays))
                spp.ensure_directory_MPI(dirOut, spp.one_hour)
                FRP_2_map_LU_split(land_use_loc, grid_loc[0],
#                               'e:\\results\\fires\\IS4FIRES_v3_0_VIIRS\\glob_3_0_LST_daily\\IS4FIRES_v3_0_glob_3_0_%Y%m%d_LST_daily.nc4', 
                               'e:\\results\\fires\\IS4FIRES_v3_0_grid_FP_2024\\glob_3_0_LST_daily\\IS4FIRES_v3_0_glob_3_0_%Y%m%d_LST_daily.nc4', 
                               startDay, dt.datetime(2020,12,31), stepDays * spp.one_day,  
#                               startDay, dt.datetime(2017,12,31), stepDays * spp.one_day,  
#                               startDay, dt.datetime(2024,11,8), stepDays * spp.one_day,  
                               dirOut, 'FRP_LU_wise_VIIRS_%s_start%s_step%id.nc4' % 
                               (grid_loc[1], startDay.strftime('%Y%m%d'), stepDays),
                               mpisize_loc, mpirank_loc, comm)

