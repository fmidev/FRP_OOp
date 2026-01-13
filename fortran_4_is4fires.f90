!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! 
! This file includes a bunch of FORTRAN subroutines for IS4FIRES
!
! Compile it with something like:
!
! subprocess.run(["python", "-m", "numpy.f2py", "-c", "-m", "FORTRAN_4_IS4FIRES", "<this FORTRAN file>"])
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
subroutine points_to_granule(x3d, y3d, z3d, &               ! granule, (0:nx-1, 0:ny-1)
                           & xp3d_ar, yp3d_ar, zp3d_ar, &   ! points, (0:nPoints-1)
                           & idxOut, &                      ! output coordinates (0:1, 0:nPoints-1)
                           & nx, ny, nPoints)               ! sizes
  !
  ! Does the bulky work of projecting a bunch of points to the granule
  ! Note that both the granule and the points are represented in Cartezian grid
  !
  implicit none
  !
  ! Imported parameters
  real*4, dimension(0:nx-1, 0:ny-1), intent(in) :: x3d, y3d, z3d
  real*4, dimension(0:nPoints-1), intent(in) :: xp3d_ar, yp3d_ar, zp3d_ar
  integer*4, dimension(0:1,0:nPoints-1), intent(out) :: idxOut
  ! sizes must be at the end
  integer*4, intent(in) :: nx, ny, nPoints
  !
  ! Local variables
  integer :: iXc, iYc, iP, ixD, iyD, itmp, jtmp, iSearch, iXcNew, iYcNew
  real :: xNew, yNew   ! use as a hint for search of the next point
  real :: xp3d, yp3d, zp3d
  real :: x0, x1, y0, y1, z0, z1, d
  logical :: done, printit, ifGotBetter !, ifBigError
  integer, parameter :: range2check = 20
  real, dimension(-range2check:range2check,-range2check:range2check) :: cost
  character(len=7) :: sTmp
  character(len=200) :: str
  
  printit = .FALSE.
  
  !
  ! nullify output indices and prepare to the cycle
  idxOut(:,:) = -1
  xNew = -1
  yNew = -1
  do iP = 0, nPoints-1
    !
    ! Get initial guess: center, corners, previous value
    !
    iXc = -1
    iYc = -1
    d  = 1.0            ! Square euclidian distance in units of earth_raduis^2, big enough to start with
    xp3d = xp3d_ar(iP)  ! read the point coordinates into local var - simpler code
    yp3d = yp3d_ar(iP)
    zp3d = zp3d_ar(iP)
    if(printit)then
      print *,''
      print *, 'Starting new point:', iP
    endif

    if (xNew > -0.5 .and. xNew < nx - 0.5 .and. yNew > -0.5 .and. yNew < ny - 0.5 ) then
      ! try previous
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, nint(xNew), nint(yNew), d, ifGotBetter, nx, ny)
      if (printit) print *, "prev ", ixC, iYc, "current point ", iP !!!, lons(ixC, iYC), lats(ixC, iYC))
    endif
    
    if (abs(d) > 1e-2) then  !! Might be grave wrong, check the grid center and corners
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, nx/2, ny/2, d, ifGotBetter, nx, ny)
      if (printit) print *, "C ", ixC, iYc   !, lons(ixC, iYC), lats(ixC, iYC))
      
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc,    0,    0, d, ifGotBetter, nx, ny)
      if (printit) print *, "c1 ", ixC, iYc  ! lons(ixC, iYC), lats(ixC, iYC))
      
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc,    0,   ny-1, d, ifGotBetter, nx, ny)
      if (printit) print *, "c2 ", ixC, iYc  ! ), lons(ixC, iYC), lats(ixC, iYC))
      
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc,   nx-1,    0, d, ifGotBetter, nx, ny)
      if (printit) print *, "c3 ", ixC, iYc   ! ), lons(ixC, iYC), lats(ixC, iYC))
      
      call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc,   nx-1,   ny-1, d, ifGotBetter, nx, ny)
      if (printit) print *, "c4 ", ixC, iYc   ! ), lons(ixC, iYC), lats(ixC, iYC))
    endif
    !
    ! For the case something is completely wrong
    !
    if(d >= 1.0)then
      if(printit) print *,'Something is completely wrong: d=',d, ', skip the point'
      idxOut(0, iP) = -1
      idxOut(1, iP) = -1
      cycle
    endif
    do iSearch = 1, 500
      !
      ! Let's allow for existence of local minima. Search will be repeated up to 10 times.
      ! After each search, an area around the suggested point is checked for a better location
      ! If found, this location becomes the next start of the search
      !
      do ixD = 1, max(nx,ny)
        !
        ! Check single-direction movements
        ! iTmp is a dummy index, just to ensure that we do not get out of the gtid
        !
        done=.true.  ! improvement in any direction puts it to false
  
        do itmp = iYc, ny-2      ! Up
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, iXc, iYc+1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Up ", ixC, iYc   !), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo
        do itmp = iYc, 1, -1     !Down
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, iXc, iYc-1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Do ", ixC, iYc  ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo

        ! For lon-global grids, wrapping should be allowed, for non-wrapped grids modulo/metrics 
        ! will ensure that we do not stick out of the grid 
        ! iTmp is dummy index, just to ensure that we do not get stuck in the loop if something goes wrong
        !
        do itmp = 0, nx-1         ! Right
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc, nx-1)+1, iYc, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Ri ", ixC, iYc   ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo
        do itmp = nx-1, 1, -1     ! Left
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc-2,nx-1)+1, iYc, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Le ", ixC, iYc  ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo
        !
        ! What about diagonal steps?
        !
        do itmp = 1, min(ny-2 - iYc, nx-1)       ! Up-right
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc, nx-1)+1, iYc+1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Up-ri ", ixC, iYc   !), lons(ixC, iYC), lats(ixC, iYC))
            done=.false.
        enddo
        do itmp = 1, min(iYc, nx-1)  ! Down-right
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc, nx-1)+1, iYc-1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Do-ri ", ixC, iYc  ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo

        ! For lon-global grids, wrapping should be allowed, for non-wrapped grids modulo/metrics 
        ! will ensure that we do not stick out of the grid 
        ! iTmp is dummy index, just to ensure that we do not get stuck in the loop if something goes wrong
        !
        do itmp = 1, min(ny-2 - iYc, nx-1)              ! Up-left
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc-2,nx-1)+1, iYc+1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Up-le ", ixC, iYc   ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo
        do itmp = 1, min(iYc, nx-1)          ! Down-Left
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXc, iYc, modulo(iXc-2,nx-1)+1, iYc-1, d, ifGotBetter, nx, ny)
          if (.not. ifGotBetter) exit
          if (printit) print *, "Do-le ", ixC, iYc  ! ), lons(ixC, iYC), lats(ixC, iYC))
          done=.false.
        enddo
      
        if (done) exit
      enddo  ! number of cycles

      if (ixD >= max(nx,ny)) then
        !! Something went severely wrong: normally much less iterations needed
        print *, "Could not project a point to anygrid"
      endif

      !Find next nearest in each direction from the nearest one
      if (ixC == nx-1) then
        ixD=-1
      elseif (ixC == 0) then
        ixD=1
      else
        d = d3(x3d(iXc+1, iYc),xp3d , y3d(iXc+1, iYc),yp3d ,z3d(iXc+1, iYc), zp3d)
        if (d > d3(x3d(iXc-1, iYc),xp3d , y3d(iXc-1, iYc),yp3d ,z3d(iXc-1, iYc), zp3d)) then
          ixD=-1
        else
          ixD=1
        endif
      endif 

      if (iyC == ny-1) then
        iyD=-1
      elseif (iyC == 0) then
        iyD=1
      else
        d = d3(x3d(iXc, iYc+1),xp3d , y3d(iXc, iYc+1),yp3d ,z3d(iXc, iYc+1), zp3d)
        if (d > d3(x3d(iXc, iYc-1),xp3d , y3d(iXc, iYc-1),yp3d ,z3d(iXc, iYc-1), zp3d)) then
          iyD=-1
        else
          iyD=1
        endif
      endif

      !! Make inter/extrapolation
      !! r0 Vector from nearest gridpoint to the target
      x0 = xp3d - x3d(iXc, iYc) 
      y0 = yp3d - y3d(iXc, iYc) 
      z0 = zp3d - z3d(iXc, iYc) 

      ! vector from nearest to second nearest in x grid direction
      x1 =  x3d(iXc + ixD, iYc) - x3d(iXc, iYc)
      y1 =  y3d(iXc + ixD, iYc) - y3d(iXc, iYc)
      z1 =  z3d(iXc + ixD, iYc) - z3d(iXc, iYc)

      !nearest point + projection of r0 onto the vector
      xNew = ixC + ixD * (x0*x1+y0*y1+z0*z1)/(x1*x1+y1*y1+z1*z1) 

      ! vector from nearest to second nearest in y grid direction
      x1 = x3d(iXc, iYc + iyD) - x3d(iXc, iYc) 
      y1 = y3d(iXc, iYc + iyD) - y3d(iXc, iYc) 
      z1 = z3d(iXc, iYc + iyD) - z3d(iXc, iYc) 

      !nearest point + projection of r0 onto the vector
      yNew = iyC + iyD * (x0*x1+y0*y1+z0*z1)/(x1*x1+y1*y1+z1*z1) 

!     if(printit)then
!           !!! interpolated lon and lat
!           x1=fu_2d_interpolation (pAnyGrdParam(iAg)%xC, xNew,yNew, nx,ny, linear, nearestPoint)
!           y1=fu_2d_interpolation (pAnyGrdParam(iAg)%yC, xNew,yNew, nx,ny, linear, nearestPoint)
! 
!           x0 = abs(x1-flon) 
!           if (x0 > 355.) x0 = abs(modulo(x1-flon+180., 360.) - 180.)
! 
!           x0 = x0 * cos(flat*degrees_to_radians) 
!           y0 = abs(y1-flat)     
! 
!           ! fu_2d_interpolation does not extrapolate beyond the ceners mesh,
!           ! while we continue the edge gradient infinitely, so the ceck 
!           ! should work only within the centers mesh
! 
!           ! out of grid: do not force big error  
!           if ((xnew < 1.     .and. ixc == 1)  .or. &
!             & (xNew > nx  .and. ixc ==nx)  .or. &
!             & (ynew < 1.     .and. iyc == 1)  .or. &
!             & (yNew > ny  .and. iyc == ny) ) then
!             x0 = -1.
!            y0 = -1.
!           endif
! 
!           ifBigError =  .not. (x0 < eps_degrees .and. y0 < eps_degrees ) 
!           if (ifBigError) then
!             call msg_warning("Too large error in reprojection to anygrid", sub_name)
!             printit = .true.
!           endif
! 
!           if (printit) then
!             call msg("lon,              lat", fLon, fLat)
!             call msg("lon-interp, latinterp", x1, y1)
! 
!             call msg("absolute error error in m", x0*1.1e5, y0*1.1e5)
! 
!             iTmp = iXc + (iYc-1)*nx !! 1D grid index
!             call msg("relative error in grid cells", & 
!                  & x0*1.1e5 / pAnyGrdParam(iAg)%dx(iTmp),  y0*1.1e5/ pAnyGrdParam(iAg)%dy(iTmp))
!             call msg("")
! 
!             call msg("Found nearest cell ", ixC, iyC)
!             call msg("xNew, yNew", xNew, yNew)
! 
!             iTmp = max(1,ixC-1)
!             jTmp = min(nx,ixC+1)
! 
!             call msg ("lons (ixmin,ixmax):", iTmp, jTmp)
!             if (iyC>1) call msg("lons-1:", lons(itmp:jtmp, iyC-1))
!               call msg("lons 0:", lons(itmp:jtmp, iyC))
!             if (iyC<ny)  call msg("lons+1:", lons(itmp:jtmp, iyC+1))
!             call msg ("lats (ixmin,ixmax):", iTmp, jTmp)
!             if (iyC>1)call msg("lats-1:", lats(itmp:jtmp, iyC-1))
!            call msg("lats 0:", lats(itmp:jtmp, iyC))
!             if (iyC<ny) call msg("lats+1:", lats(itmp:jtmp, iyC+1))
!
!             call ooops("")
!           endif
!           if (ifBigError) then
!             call set_error("Too large error in reprojection to anygrid", sub_name)
!          endif
!      endif    ! debug print
      !
      ! Store the found index into the output array
      !
      idxOut(0, iP) = iXc
      idxOut(1, iP) = iYc

      ! check around the point found: discretization can play dirty tricks
      !
      d = d3(x3d(iXc,iYc),xp3d, y3d(iXc,iYc), yp3d, z3d(iXc,iYc), zp3d)
      cost = d
      iXcNew = iXc
      iYcNew = iYc
      do itmp = -range2check, range2check
        do jtmp = -range2check, range2check
          call tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, iXcNew, iYcNew, &
                   & min(nx-1,max(0,iXc+itmp)), min(ny-1,max(0,iYc+jtmp)), cost(itmp,jtmp), &
                   & ifGotBetter, nx, ny)
          if(ifGotBetter)then
            if(printit) print *,'Better option:', iXcNew, iYcNew, cost(itmp,jtmp)
            exit
          endif

!          cost(itmp,jtmp) = d3(x3d(min(nx-1,max(0,iXc+itmp)),min(ny-1,max(0,iYc+jtmp))),xp3d, &
!          & y3d(min(nx-1,max(0,iXc+itmp)),min(ny-1,max(0,iYc+jtmp))), yp3d, &
!          & z3d(min(nx-1,max(0,iXc+itmp)),min(ny-1,max(0,iYc+jtmp))), zp3d)
!          ! store the locaiton and cost
!          if(cost(itmp,jtmp) < d)then
!            d = cost(itmp,jtmp)
!            iXcNew = min(nx-1,max(0,iXc+itmp))   ! iXc + itmp
!            iYcNew = min(ny-1,max(0,iYc+jtmp))   ! iYc + jtmp
!            if(printit) print *,'Better option:', iXcNew, iYcNew, d
!          endif
        end do
        if(ifGotBetter)exit   ! found improvement - go for it, no need to check rest of the square
      end do
      !
      ! If cost(0,0) is the lowest, we are done. Otherwise repeat the search starting from (iXc,iYc)
      !
      cost(:,:) = cost(:,:) / minval(cost)
      if(cost(0,0) == 1.0)then
        if(printit) print '(A,2i5,2x,A,i5)', 'Global (wide-local) optimum found at:', iXc, iYc, 'Point: ', iP
        exit   ! this point has been projected
      else
        ! found a better point in the vicinity: repeat the search starting from it
        if(printit)then
          print '(A25, 2i5, A20, F10.7)', 'Local optimum found at:', iXc, iYc, 'its relative cost:', cost(0,0)
          print *, 'Better location:', minloc(cost)-range2check-1, iXcNew, iYcNew  ! minloc does not check the range, just size
          str = '     '
          do jtmp = -range2check, range2check
            write(sTmp, fmt='(i7)')jtmp
            str = trim(str) // sTmp
          end do
          print '(200A)', trim(str)
          do itmp = -range2check, range2check
            print '(i3,2x 100(f6.3,1x))', itmp, cost(iTmp,:)
          end do    
          print *, 'Redo the search, round ', iSearch+1
        endif  ! prrintit
        iXc = iXcNew
        iYc = iYcNew
      endif  ! if the found point is the best
    end do  ! cycle over the search rounds
  end do  ! points


  CONTAINS
  
    real function d3(x1,x2,y1,y2,z1,z2) 
      implicit none
      real, intent(in) :: x1,x2,y1,y2,z1,z2
      d3 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
    end function d3

    subroutine tryD3(x3d,y3d,z3d, xp3d,yp3d,zp3d, ixC, iyC, ixtry, iytry, cost, ifGotBetter, nx, ny)
      real, dimension(0:nx-1,0:ny-1), intent(in) :: x3d,y3d,z3d !!grid
      real, intent(in) :: xp3d,yp3d,zp3d  !! point
      integer, intent(in) :: ixtry, iytry  !! Trial
      integer, intent(inout) :: ixC, iyC   !! Adjust if got better
      real, intent(inout) :: cost
      logical, intent(out) :: ifGotBetter
      integer :: nx, ny

      ! local variable
      real :: prevcost

      prevcost = cost

      cost = d3(x3d(ixtry, iytry),xp3d , y3d(ixtry, iytry),yp3d ,z3d(ixtry, iytry), zp3d)
      ifGotBetter = (prevcost == -1 .or. cost < prevcost) 

      if (ifGotBetter) then
        ixC = ixTry
        iyC = iyTry
        if (printit) print *, "Better cost:", ixTry, iyTry, cost 
      else
        if (printit) print *, "Worse cost:", ixTry, iyTry, cost 
        cost = prevcost
      endif
    end subroutine tryD3

end subroutine points_to_granule

