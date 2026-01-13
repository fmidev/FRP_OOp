'''
Created on 23.5.2022

This is the basic class for handling the satellite granules. Most of functions are
not implemented 

Functions declared:
- __init__:   Sets abasic file names 
- get_pixel_size:  basic MODIS geomatry
- full_resol_geo_from_coarse: Takes the so-called reduced-resolution fields of longitude
                              and latitude and makes full-resolutionn ones
- unpack_MxD03_byte_1: Bit fields within each byte are numbered from the left: 7, 6, 5, 4, 3, 2, 1, 0.
- unpack_MxD14_QA:     unpacks the quality field. NotImplemented
- pick_granule_data_IS4FIRES_v3_0: gets the needed data from MxD14 and MxD35
######- detection_limit;     returns the detection limit map as a function of the day/night flag and pixel size
- pick_granule_data_IS4FIRES_v2_0: gets data needed for old IS4FIRES version from MOD14
- write_granule_IS4FIRES_v2_0:  write the granule down as IS4FREIS 2.0 need
- draw_granule:        Draws area covered by this swath, with a bit free space around:


@author: sofievm
'''

import numpy as np
import numpy.f2py
import os, shutil, glob, subprocess
from toolbox import supplementary as spp
import matplotlib as mpl
import datetime as dt
from toolbox import drawer, gridtools
from scipy.interpolate import griddata

R_Earth = 6378.137  # Earth radius, km
#R_Earth = 6371


ifFortranOK = False
try:
    import fortran_4_is4fires
    ifFortranOK = True
except:
    try:
        from src import fortran_4_is4fires
        ifFortranOK = True
    except: pass

    # Compile the library and, if needed, copy it from the subdir where the compiler puts it
    # to the current directory
    #
    if not ifFortranOK: 
# For embedded code and working f2py.compile(), this one is good:
#        vCompiler = np.f2py.compile(strFortran_4_is4fires, modulename='fortran_4_is4fires', 
#                                    verbose=1, extension='.f90')
        # via external call
        outcome = subprocess.run(["f2py3", "-c", "-m", "fortran_4_is4fires", "fortran_4_is4fires.f90"], 
                                 check=True, shell=True)
        if outcome.returncode != 0: raise ValueError('f2py error')
#        fortran_compiler = mesonpy.get_compiler('fortran')
#        my_fortran_lib = fortran_compiler.compile('FORTRAN_4_IS4FIRES.f90', 'fortran_4_is4fires') #, # Output base name
##                                                  install: True, # Install the library
##                                                  install_dir: get_option('prefix') / get_option('libdir')
#        subprocess.run(["f2py", "-c", "-m", "fortran_4_is4fires", "c:\\ap\\eclipse\\IS4FIRES_v3_0\\src\\fortran_4_is4fires.f90"], 
#                       check=True, shell=True) #, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cwd = os.getcwd()     # current working directory
        if os.path.exists(os.path.join('fortran_4_is4fires','.libs')):
            list_of_files = glob.glob(os.path.join('fortran_4_is4fires','.libs','*'))
            latest_file = max(list_of_files, key=os.path.getctime)
            shutil.copyfile(latest_file, os.path.join(cwd, os.path.split(latest_file)[1]))
        try: 
            from src import fortran_4_is4fires
            ifFortranOK = True
        except:
            try:
                import fortran_4_is4fires
                ifFortranOK = True
            except:
                ifFortranOK = False

if not ifFortranOK:
    print('>>>>>>> FORTRAN failed, cannot do much. Projection of pixels on granule is disabled')





#################################################################################
#
#  A void class made for storing named fields after unpacking them from the 
#  granule bit-packed variables
#
#################################################################################

class unpacked_cloud_fields():
    def __init(self):
        return

#################################################################################
#
# Class granule_basic. Mostly, a placeholder
#
#################################################################################

class granule_basic():
    
    def __init__(self, satellite, now_UTC, chFRPfilesTempl, chAuxilFilesTempl, log, ifSilent=False):
        self.type = 'void'
        self.satellite = satellite
        self.now_UTC = now_UTC
        self.templFRPFiles = chFRPfilesTempl        # ready for now.strftime(template)
        self.templAuxilFiles = chAuxilFilesTempl
        self.ifSilent = ifSilent
        self.log = log

    #=============================================================
    def get_FRP_FNm(self):
        return self.templFRPFiles.strftime(self.now_UTC)
    
    #=============================================================
    def get_geo_FNm(self):
        return self.templAuxilFiles.strftime(self.now_UTC)

    #=============================================================
    def now(self): return self.now_UTC
    
    #=============================================================
    def get_pixel_size(self):
        raise NotImplemented


    #=================================================================

    def resample_field_with_griddata(self, fldIn, ifLongitude, chMethod, scanSzIn_track,
                                     pixel_size_scaling):
        #
        # PROBLEMATIC: cannot extrapolate. Must be supplied with fields with all borders defined
        #
        # Jumps between resolutions of the field. Works for both up- and 
        # down-sampling but needs information on scan sizes: its width along track
        # and its length along scan (32 x 6400 for VIIRS or 10x1354 for MODIS)
        # If the field is longitude, takes care of the -180 : 180 jump
        # Methods: nearest, linear, cubic
        #
        nScans = round(fldIn.shape[0] / scanSzIn_track)
        scanSzOut_track = round(scanSzIn_track / pixel_size_scaling) # swath along track is a bunch of scans
        scanSzOut_scan = round(fldIn.shape[1] / pixel_size_scaling)  # scan takes the whole 1-st dimension
        fldOut = np.ones(shape=(scanSzOut_track * nScans, scanSzOut_scan),
                         dtype = fldIn.dtype) * np.nan
        # make the input and output coordinates
        if pixel_size_scaling > 1:
            # change to coarser grid, i.e. smaller array: downsampling
            # input points coordinates are trivial
            gridIn = np.mgrid[0:scanSzIn_track, 0:fldIn.shape[1]]
            # coordinates of output points in the input grid
            gridOut_x, gridOut_y = np.mgrid[pixel_size_scaling / 2. - 0.5 : scanSzOut_track 
                                            : pixel_size_scaling,
                                            pixel_size_scaling / 2. - 0.5 : scanSzOut_scan 
                                            : pixel_size_scaling]
        else:
            gridIn = np.mgrid[1./pixel_size_scaling / 2.- 0.5 : scanSzOut_track : 1./pixel_size_scaling,
                              1./pixel_size_scaling / 2.- 0.5 : scanSzOut_scan : 1./pixel_size_scaling]
            gridOut_x, gridOut_y = np.mgrid[0:scanSzOut_track, 0:scanSzOut_scan]
        # The actual inter / extrapolation
        points = gridIn.reshape((2,-1)).T
        for iScan in range(nScans):
            fldOut[iScan * scanSzOut_track : (iScan + 1) * scanSzOut_track,
                   :] = griddata(points, 
                                 fldIn[iScan * scanSzIn_track : 
                                       (iScan + 1) * scanSzIn_track].ravel(), 
                                 (gridOut_x, gridOut_y), method=chMethod)
        return fldOut


    #=============================================================

    def downsample_field(self, DS_factor_track, DS_factor_scan, scanSz, fldIn, method='mid4'):
        #
        # Downsamples the given field using either of the two methods: picking 4 points or making a mean over a range
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
            ## Faster equivalent to the MAS method
            sls = slice(DS_factor_scan//2-1,DS_factor_scan//2+1)
            slt = slice(DS_factor_track//2-1,DS_factor_track//2+1)
            f_av = np.mean(fIn[:,:,slt,:,sls], axis=(2,4))
        elif method == 'median': 
            ## marginally better compression (1-2%), a some 5% more expensive
            f_av = np.mean(fIn[:,:,:,:,:], axis=(2,4))
        else:
            raise ValueError("Unknown method (\"%s\") can be one of mid4 or median"%(method))
        
        return f_av.reshape(f_av.shape[0] * f_av.shape[1], f_av.shape[2])


    #========================================================================

    def draw_pixel_size(self, chOutFNm, ifSilent=False, ifOverlap_map=False):
        #
        # Draws the basic features of the granule: pixel sizes, areas, overlaps, etc.
        #
        histogr, bin_edges = np.histogram(self.area,bins=100)
        histCum = np.cumsum(histogr)
        fig = mpl.pyplot.figure(figsize=(9,10))
        gs = fig.add_gridspec(2,2)
        ax0 = fig.add_subplot(gs[0,0])
        axt0 = ax0.twinx()
#        axt0.plot(range(self.N_NA), self.theta_NA * 180 / np.pi, label='theta, %g\xb0..%g\xb0' % 
#                        (np.min(self.theta_NA * 180 / np.pi), np.max(self.theta_NA * 180 / np.pi)),
#                        color='blue')
#        ax0.plot(range(self.N_NA), self.dS_NA, label='size along scan, %5.3f..%5.3f km' % 
#                       (np.min(self.dS_NA), np.max(self.dS_NA)), color='violet',linewidth=5)

        axt0.plot(range(self.N), self.theta * 180 / np.pi, label='theta, %5.2f\xb0...%5.2f\xb0' % 
                        (np.min(self.theta * 180 / np.pi), np.max(self.theta * 180 / np.pi)),
                        color='cornflowerblue')
        ax0.plot(range(self.N), self.dS, label='size along scan, %4.2f...%4.2f km' % 
                       (np.min(self.dS), np.max(self.dS)), color='orange')
        ax0.plot(range(self.N), self.dT, label='size along track, %4.2f...%4.2f km' % 
                       (np.min(self.dT), np.max(self.dT)), color='green')
        ax0.plot(range(self.N), self.area, label='pixel area, %4.2f...%4.2f km2' % 
                 (np.min(self.area), np.max(self.area)), color='red')
        ax0.set_xlabel('pixel nbr')
#        ax0.set_ylabel('pixel area [km2],     pixel size [km]')
        drawer.multicolor_label(ax0, ['pixel area [km2],     ', 'pixel', 'size [km]'],
                             ['r','orange','green'], 'y', anchorpad=-1)
        axt0.set_ylabel('theta [deg]', color='b')
        lines, labels = axt0.get_legend_handles_labels()
        lines2, labels2 = ax0.get_legend_handles_labels()
        axt0.legend(lines + lines2, labels + labels2, loc=9, fontsize=10)
#        ax0.set_ylim(0,1)
        axt0.set_ylim(-65,65)
        ax0.grid()
        ax0.set_title(self.type + ' pixel vs swath angle')
        #
        # histogram: cases vs area
        #
        ax1 = fig.add_subplot(gs[1,0])
        axt1 = ax1.twinx()
        ax1.plot(bin_edges[:-1], histogr, label='area histogram', color='b')
        axt1.plot(bin_edges[:-1], histCum, label='area cumul.histogr', color='r')
#        axes[1].plot(bin_edges[:-1], histogr / np.max(histogr), label='area distr.density, normed')
#        axes[1].plot(bin_edges[:-1], histogr * (bin_edges[1]-bin_edges[0]), label='area distr.density')
#        axes[1].plot(bin_edges[:-1], histCum / histCum[-1], label='area distr.function')
        ax1.set_xlabel('pixel area, km2')
#        ax1.set_xlim(0,0.8)
        ax1.set_ylabel('number of pixels', color='b')
        axt1.set_ylabel('number of pixels, cumul.',color='r')
        lines, labels = axt1.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc=5, fontsize=10)
        ax1.grid()
        ax1.set_title(self.type + ' pixel area histogram')
        #
        # histogram: area fraction vs area
        #
        areaFract = histogr * (bin_edges[:-1] + bin_edges[1:]) * 0.5   # area covered by these pixels
        cumArea = np.cumsum(areaFract)
        ax2 = fig.add_subplot(gs[0,1])
        axt = ax2.twinx()
#        ax2.plot(bin_edges[:-1], areaFract, label='area fraction histogram')
        ax2.plot(bin_edges[:-1], cumArea, label='cumulative area', color='blue')
        ax2.set_xlabel('pixel area, km2')
        ax2.set_ylabel('covered area, km2', color='b')
#        ax2.set_xlim(0,0.8)
        ax2.set_ylim(0,cumArea[-1] * 1.05)
        ax2.grid()
        axt.plot(bin_edges[:-1], cumArea / cumArea[-1], color='blue')
        axt.set_ylim(0,1.05)
        axt.set_ylabel('fraction of covered area',color='blue')
        axt.grid(color='cornflowerblue') #'royalblue')# 'lightblue')
        ax2.set_title(self.type + ' histogram of single-scan-line area')
        #
        # Scan lines overlap
        #
        ax3 = fig.add_subplot(gs[1,1])
        axt2 = ax3.twinx()
        axt2.plot(range(self.N), self.dist_nadir, label='distance from nadir',color='blue')
        axt2.set_ylabel('distance from nadir point [km]', color='b')
        ax3.plot(range(self.N-1), self.overlap_S * 100, label='overlap along scan',color='orange')
        ax3.plot(range(self.N), (self.overlap_T *100) ,label='overlap along track',color='green')
        ax3.set_xlabel('pixel nbr')
#        ax3.set_ylabel('overlap, %')
        drawer.multicolor_label(ax3, ['              over', 'lap, %'],
                                ['orange','green'], 'y', anchorpad=-1)
        ax3.legend(loc=2)
        ax3.grid()
        ax3.set_title(self.type + ' overlap of sequential scans')
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = axt2.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc=9, fontsize=10)

        mpl.pyplot.figtext(0.03, 0.96,'a)', fontsize=14)
        mpl.pyplot.figtext(0.52, 0.96,'b)', fontsize=14)
        mpl.pyplot.figtext(0.03, 0.46,'c)', fontsize=14)
        mpl.pyplot.figtext(0.52, 0.46,'d)', fontsize=14)

        mpl.pyplot.subplots_adjust(left=0.08, bottom=0.05, right=0.91, top=0.95, 
                                   wspace=0.45, hspace=0.25)
        mpl.pyplot.savefig(chOutFNm,dpi=500)
        mpl.pyplot.clf()
        mpl.pyplot.close()

        if not ifSilent: print(chOutFNm)
        
        #
        # overlap pattern
        #
        if ifOverlap_map:
            fig = mpl.pyplot.figure(figsize=(10,3))
            cbFract = 0.1   #0.04
            cbPad = 0.22   # 0.055 
            cbAspect = 100  #  38
            ax = fig.add_subplot()
            cs = ax.imshow(self.overlap_pattern.T, aspect='auto') # [:,::10])
            cbar = fig.colorbar(cs,orientation='horizontal', ax=ax, # location='bottom',
                                fraction=cbFract, pad=cbPad, aspect=cbAspect) #, pad="5%")
            ax.set_title('Overlap: number of observations by sequential scans')
            ax.set_xlabel('Pixel along scan')
            ax.set_ylabel('Pixel along track')
            if chOutFNm.find('.png') > 0:
                mpl.pyplot.savefig(chOutFNm.replace('.png','_overlap.png'),dpi=500)
            else:
                mpl.pyplot.savefig(chOutFNm + '_overlap.png',dpi=500)
            mpl.pyplot.clf()
            mpl.pyplot.close()


    #=============================================================
    
    def project_points_to_granule(self, lonPoints, latPoints):
        #
        # Projects a set of points given in geographical coordinates to this granule
        #
        if not ifFortranOK:
            raise NotImplemented('FORTRAN for point_to_granule not OK, python is not implemented')
#        print('Projecting points to granule...')
        #
        # The work will be done in x-y-z Cartesian space, in meters, so integers are fine
        #
        granX, granY, granZ = gridtools.xyzFromLonLat(self.lon, self.lat)
        minX = np.min(granX)
        maxX = np.max(granX)
        minY = np.min(granY)
        maxY = np.max(granY)
        minZ = np.min(granZ)
        maxZ = np.max(granZ)
        
        ifPrepare_input_F = False
        
        # same for fires
        pointX, pointY, pointZ = gridtools.xyzFromLonLat(lonPoints, latPoints)
        
        # points that can be inside the granule, not necessarily though
        ifPointsNearOrInGran = ((pointX >= minX) & (pointX <= maxX) & 
                                   (pointY >= minY) & (pointY <= maxY) & 
                                   (pointZ >= minZ) & (pointZ <= maxZ)) #.astype(np.int8)
        nPoints = np.sum(ifPointsNearOrInGran)
        
        # anything left?
        if nPoints == 0:
#            print(self.now_UTC.strftime('No points near granule %Y%m%d-%H%M'))
            return (None, None)
#        print('Projected points to granule: ', self.now_UTC, nPoints)
        
        # A single array is returned: (2, nPoints) - ix, iy indices in the granule of nPoints points
        # output indices
        idxOut = np.ones(shape=(2, nPoints),dtype=np.int32) * (-1)

#        # output coordinates (0:1, 0:nPoints-1)
        if ifPrepare_input_F:
            print('Storing the granule')
            with open('d:\\tmp\\MODIS_gran_xyz.txt','w') as fOut:
                for ix in range(self.lon.shape[0]):
                    for iy in range(self.lon.shape[1]):
                        fOut.write('%i %i %g %g %g %g %g\n' % (ix, iy, granX[ix,iy]/gridtools.R_earth, 
                                                               granY[ix,iy]/gridtools.R_earth, 
                                                               granZ[ix,iy]/gridtools.R_earth,
                                                               self.lon[ix,iy], self.lat[ix,iy]))
            print('Storing the points that can be in it')
            with open('d:\\tmp\\fires_4_gran_xyz.txt','w') as fOut:
                for ix in range(nPoints):
                    fOut.write('%i %g %g %g %g %g\n' % (ix, pointX[ifPointsNearOrInGran][ix]/gridtools.R_earth, 
                                                        pointY[ifPointsNearOrInGran][ix]/gridtools.R_earth,
                                                        pointZ[ifPointsNearOrInGran][ix]/gridtools.R_earth,
                                                        lonPoints[ifPointsNearOrInGran][ix], 
                                                        latPoints[ifPointsNearOrInGran][ix]))
        
        idxOut[:,:] = fortran_4_is4fires.points_to_granule(granX/gridtools.R_earth, 
                                                           granY/gridtools.R_earth, 
                                                           granZ/gridtools.R_earth,    # granule, (0:nx-1, 0:ny-1)
                                                           pointX[ifPointsNearOrInGran]/gridtools.R_earth, 
                                                           pointY[ifPointsNearOrInGran]/gridtools.R_earth,
                                                           pointZ[ifPointsNearOrInGran]/gridtools.R_earth, # points, (0:nPoints-1)
                                                           granX.shape[0], granX.shape[1], nPoints)   #  nx, ny, nPoints
        #
        # Points outside the granule or wrongly projected should be flagged out
        #
        distance = (np.square(granX[idxOut[0,:],idxOut[1,:]] - pointX[ifPointsNearOrInGran]) +
                    np.square(granY[idxOut[0,:],idxOut[1,:]] - pointY[ifPointsNearOrInGran]) +
                    np.square(granZ[idxOut[0,:],idxOut[1,:]] - pointZ[ifPointsNearOrInGran]))
        # swath parameters in granules are in km
        maskFiresOutsideGran = distance > 1e6 * (np.square(self.dS[idxOut[1,:]]) + np.square(self.dT[idxOut[1,:]]))
        #
        # Preserve all informatin: the flag is 0 for points far from the granule, 1 when they are 
        # inside the granule xyz envelope and 2 when they are inside the granule and their coeefficients 
        # are calculated
        #
        flag_pointsNearOrInGran = ifPointsNearOrInGran.astype(np.int8)
        flag_pointsNearOrInGran[ifPointsNearOrInGran] += np.logical_not(maskFiresOutsideGran)
        idxOut[:,maskFiresOutsideGran] = -1

        return (flag_pointsNearOrInGran, idxOut)  # flag near/in/out, indices in granule
            


    #=============================================================
    def full_resol_geo_from_coarse(self):
        raise NotImplemented
         
    #=============================================================
    def unpack_info_byte_1(self): 
        raise NotImplemented
         
    #=============================================================
    def unpack_MxD14_QA(self):
        raise NotImplemented
         
    #=============================================================
    def pick_granule_data_IS4FIRES_v3_0(self):
        raise NotImplemented
         
    #=============================================================
    def pick_granule_data_IS4FIRES_v2_0(self):
        raise NotImplemented

    #=============================================================
    def from_nc(self):
        raise NotImplemented
    
    #=============================================================
    def write_granule_IS4FIRES_v2_0(self):
        raise NotImplemented

    #=============================================================
    def to_nc(self):
        raise NotImplemented
    
    #=============================================================
    def draw_granule(self):
        raise NotImplemented

    #=============================================================
    def make_fire_records(self):
        raise NotImplemented
    
#============================================================================================    

if __name__ == '__main__':
    gr = granule_basic('MOD', dt.datetime.now(), 'try', 'try2', spp.log('d:\\tmp\\try.log'), ifSilent=False)
    gr.lon = np.ones(shape=(4,4), dtype=np.float64) 
    gr.lat = np.ones(shape=(4,4), dtype=np.float64)
    gr.project_points_to_granule(np.ones(shape=(15)), np.ones(shape=(15))*1.2)
    
