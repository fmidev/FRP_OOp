'''

This module contains the Quality_assurance class and necessary gadgets.
It verifies the input information, eliminates persistent and suspicious fires, 
checks consistency etc.

Created on 8.9.2020

@author: sofievm
'''

import numpy as np
from toolbox import supplementary as spp, gridtools, silamfile, namelist, MyTimeVars, drawer
import netCDF4 as nc4
import os, copy, glob
import datetime as dt
from matplotlib import pyplot as plt
import process_satellites as MxD_processor
import granule_MODIS, granule_SLSTR, granule_VIIRS
#
# Satellites that we understand
#
#sources_def = ['MOD','MYD','VJ1','VNP','MxD','Vxx']

########################################################################################
#
# Flags have default values if the corresponding QA procedure is active
#
def_scan_overlap_flag  =  np.int64(1)
def_problematic_days_flag=np.int64(10)
def_sml_huge_fires_flag = np.int64(100)
def_void_fires_flag    =  np.int64(1000)
def_frequent_fires_flag = np.int64(10000)
def_glints_flag    =      np.int64(100000)
def_maxval_flag    =      np.int64(5000000)  # just to see all zeroes

########################################################################################

class QA_basic():
    
    def __init__(self, log):
        self.log = log
        self.grid = None
        self.QA_flag = np.int64(0)

    #===============================================================

    def from_control_namelist(self, nlIni):
        pass

    #===============================================================
    @classmethod
    def void(cls, log):
        #
        return cls(log)

    #===============================================================

    def to_txt(self, chFNmOut, chStringDef, chLabel, chYear, chExtras=None):
        #
        # Writes an arbitrary set of vectors to the given file
        fOut = open(chFNmOut,'w')
        # metadata
        fOut.write('LIST = parameters\n')
        fOut.write('years = %s\n' % chYear)     # string!
        fOut.write('method = %s\n' % self.method)
        fOut.write('grid_name = %s\n' % self.gridNm)
        fOut.write(self.grid.toCDOgrid().tostr() + '\n')
        fOut.write('END_LIST = parameters\n')
        # list
        fOut.write('\nLIST = %s\n' % chLabel)
        fOut.write(chStringDef + '\n')
        ix = np.mod(self.i1d, self.grid.nx)
        iy = ((self.i1d-ix) / self.grid.nx).astype(int)
        for i, i1d in enumerate(self.i1d):
            if chExtras is None:
                fOut.write('%s = %i %i %i %g %g %i %g\n' %
                           (chLabel, i1d, ix[i], iy[i], self.lons[i], self.lats[i], self.nFires[i], self.frps[i]))
            else:
                fOut.write('%s = %i %i %i %g %g %i %g %s\n' %
                           (chLabel, i1d, ix[i], iy[i], self.lons[i], self.lats[i], self.nFires[i], self.frps[i], chExtras[i]))
        fOut.write('\nEND_LIST = %s\n' % chLabel)
        fOut.close()
        
    
    #===============================================================

    def to_kml(self, chFNmOut, chName, abbrev, chYear, chNameExtra=None, chExtra=None):
        #
        # Write a basic list of lons, lats, and nFires into a kml file
        # Note chYear is a string to allow for more than one year to be written as a label
        #
        fOut = open(chFNmOut, 'w')
        fOut.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fOut.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
        fOut.write('<Folder>\n')
        fOut.write('    <name>%s %s</name>\n' % (self.gridNm, chName))
        fOut.write('    <open>1</open>\n')
        fOut.write('    <Style>\n')
        fOut.write('        <ListStyle>\n')
        fOut.write('            <listItemType>check</listItemType>\n')
        fOut.write('            <ItemIcon>\n')
        fOut.write('                <state>open</state>\n')
        fOut.write('                <href>:/mysavedplaces_open.png</href>\n')
        fOut.write('            </ItemIcon>\n')
        fOut.write('            <ItemIcon>\n')
        fOut.write('                <state>closed</state>\n')
        fOut.write('                <href>:/mysavedplaces_closed.png</href>\n')
        fOut.write('            </ItemIcon>\n')
        fOut.write('            <bgColor>00ffffff</bgColor>\n')
        fOut.write('            <maxSnippetLines>2</maxSnippetLines>\n')
        fOut.write('        </ListStyle>\n')
        fOut.write('    </Style>\n')
#        fOut.write('    <Style id="sh_red-pushpin">\n')
#        fOut.write('        <IconStyle>\n')
#        fOut.write('            <Icon>\n')
#        fOut.write('                <href>http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png</href>\n')
#        fOut.write('            </Icon>\n')
#        fOut.write('        </IconStyle>\n')
#        fOut.write('    </Style>\n')
        for iIdx, lon in enumerate(self.lons):
            fOut.write('    <Placemark>\n')
            fOut.write('        <name>%s_%s_n%g</name>\n' % (abbrev, chYear, self.nFires[iIdx]))
            fOut.write('        <description>\n')
            fOut.write('        <![CDATA[\n')
            ix = np.mod(self.i1d[iIdx],self.grid.nx)
            fOut.write('        lon-lat coord: (%g, %g)<br>year(s)=%s, nFires=%g<br>idx1D=%i, ix=%i, iy=%i<br>\n' % 
                       (lon, self.lats[iIdx], chYear, self.nFires[iIdx], self.i1d[iIdx], ix, int((self.i1d[iIdx]-ix)/self.grid.nx)))
            if chNameExtra is None:
                fOut.write('        FRP tot=%g\n' % self.frps[iIdx])
            else:
                fOut.write('        FRP tot=%g, %s=%s\n' % (self.frps[iIdx], chNameExtra, chExtra[iIdx]))
            fOut.write('        ]]>\n') 
            fOut.write('        </description>\n')
            fOut.write('        <Point>\n')
            fOut.write('            <coordinates>%g,%g,0</coordinates>\n' % (lon, self.lats[iIdx]))
            fOut.write('        </Point>\n')
            fOut.write('    </Placemark>\n')
        fOut.write('</Folder>\n')
        fOut.write('</kml>\n')
        fOut.close()


    #=======================================================================================

    def to_nc(self, chFNmOut, dicAttr):
        #
        # Store the given object to netcdf as a set of vectors
        #
        outF = nc4.Dataset(chFNmOut, 'w', format='NETCDF4')
        outF.createDimension("fires", len(self.ix))
        for var in [('ix','i2','x_index','', self.ix),   # var_name, type, long name, unit
                    ('iy','i2','y_index','', self.iy),   # var_name, type, long name, unit
                    ('lon','f4','longitude','degrees_east', self.lons),
                    ('lat','f4','latitude','degrees_north', self.lats),
                    ('frp','f4','FRP','MW', self.frps),
                    ('nfires','i4','Nbr of fires','', self.nFires)]:
            v = outF.createVariable(var[0], var[1], ("fires"), zlib=True, complevel=5)
            v.long_name = var[2]
            if var[3] != '' :  v.units = var[3]
            outF.variables[var[0]][:] = var[4][:]
        # global attributes
        for k in dicAttr.keys():
            outF.setncattr(k, dicAttr[k])
        # grid
        silamfile.write_grid_to_nc(outF, self.grid, 'grid_fire_records')
        outF.close()

    #===================================================================

    def report(self):
        return ''


########################################################################################

class QA_overlap_scans():
    
    def __init__(self, log):
        self.QA_flag = def_scan_overlap_flag   # active QA - initiated always f the object is initiated
        self.log = log

    #===============================================================

    def from_control_namelist(self, nlIni):
        pass

    #===============================================================
    @classmethod
    def void(cls):
        #
        return cls(None)

    #===============================================================
    @classmethod
    def from_nc(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        # Retrieve parameters and initiate
        try: 
            return cls(log)
        except: return cls.void()

    #===============================================================
    
    def scale_overlapping_fires(self, i_line, i_sample, FRP, satellite):
        #
        # Satellites scans overlap towards the larger scan angles. If a fire occurs
        # in the overlapping region, it will be reported twice with two sequential
        # scans. These cases must be removed by taking average of the duplicate FRPs,
        # which may be different.
        #
        # Need the satellite to know whose granule to look at
        #
        sat_uniq = list(set(satellite))
        for s in sat_uniq:
            if s in [b'O',b'Y']:
                gran = granule_MODIS().get_pixel_size()
            elif s in [b'N', b'1']:  
                gran = granule_VIIRS().get_pixel_size()
            elif s in [b'S']:  
                gran = granule_SLSTR().get_pixel_size()
            else:
                raise ValueError('Unknown satellite abbreviation:' + s)
            #
            # The indices of the pixel in the scan are enough to take decision: line and sample
            #
            idxOK = satellite == s
            FRP[idxOK] /= gran.overlap_pattern[i_sample[idxOK], i_line[idxOK]]
        
        # This is the only basic QA function, set the flag if applied
        self.QA_flag = np.int64(1)
        

    #=======================================================================================

    def to_nc(self, fOut):
        #
        # Store the given object to netcdf as a set of vectors
        #
        if self.QA_flag: fOut.setncattr('QA_overlap_flag', self.QA_flag)

    #===================================================================

    def report(self):
        if self.QA_flag == 0: return 'No scan overlap QA'
        else: return 'Scan overlap QA processed'


########################################################################################

class QA_frequent_fires(QA_basic):
    
    #===================================================================

    def __init__(self, frequent_fires_FNm_templ, chDirMetadata, log):
        #
        # Initialises the quality_assurance class instance
        #
        self.log=log
        self.current_year = np.nan
        self.gridFF = None
        self.ifNetcdf = False
        self.ifNamelist = False
        
        # void rule?
        if frequent_fires_FNm_templ is None:
            # Initialising empty frequent-fires object
            self.QA_flag = np.int64(0)
            return
        # Store the frequent fire file with path
        if chDirMetadata is None: self.frequent_fires_list_file = frequent_fires_FNm_templ
        else: self.frequent_fires_list_file = os.path.join(chDirMetadata, frequent_fires_FNm_templ)
        #
        # Read it and prepare to apply
        #
        if '%' in self.frequent_fires_list_file:
#            self.log.log('Template stored:' + self.frequent_fires_list_file)
            pass
        else:
            if not os.path.exists(self.frequent_fires_list_file):
                raise ValueError('Frequent fires list file does not exist: ' + self.frequent_fires_list_file)
#            else:
#                self.log.log('Initialising frequent-fires object from file: ' + self.frequent_fires_list_file)
            # what format? Netcdf or namelsit in the ASCII file?
            try:
                self.read_nc_data(self.frequent_fires_list_file)
#                self.log.log('Initialized from netcdf')
            except:
                self.read_namelist(self.frequent_fires_list_file)
#                self.log.log('Initialized from namelist')
        # active QA is expected
        self.QA_flag = def_frequent_fires_flag   # active QA

    #===============================================================
    @classmethod
    def void(cls):
        #
        return cls(None, None, None)

    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni, chDirMetadata):
        #
        # In the IS4FIRES v.3.0 control file frequent fires are represented
        # via a single file with the metadata inside. Can be netcdf or namelist
        #
        try:
            return cls(nlIni.get_uniq_env('frequent_fires_list_file'), chDirMetadata, spp.log(os.path.join(chDirMetadata, nlIni.get_uniq_env('QA_log'))))
        except: 
            return cls(None, None, spp.log(os.path.join(chDirMetadata, nlIni.get_uniq_env('QA_log'))))

    #===============================================================
    @classmethod
    def from_nc_QA(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        # Retrieve parameters and initiate
        try: 
            return cls(fIn.QA_frequent_fires_list_file, None, log)  # Metadata directory is stored in the nc
        except: return cls.void()

    #===============================================================
    
    def to_nc_QA(self, fOut):
        #
        # Retrieve parameters and initiate
        if self.QA_flag: fOut.setncattr('QA_frequent_fires_list_file', self.frequent_fires_list_file)

    #===================================================================
    
    def read_nc_data(self,frequent_fire_nc_FNm):
        #
        # Reads a given netcdf file with frequent fires
        #
        fIn = nc4.Dataset(frequent_fire_nc_FNm,'r')
        fIn.set_auto_mask(False)            ## Never mask
        self.FF_frp = fIn.variables['FF_frp'][:].astype(np.float32)
        self.FF_ix = fIn.variables['FF_ix'][:].astype(int)
        self.FF_iy = fIn.variables['FF_iy'][:].astype(int)
        self.FF_lat = fIn.variables['FF_lat'][:].astype(np.float32)
        self.FF_lon = fIn.variables['FF_lon'][:].astype(np.float32)
        self.FF_nFires = fIn.variables['FF_nfires'][:].astype(int)
        self.current_year = fIn.getncattr('FF_year')
        self.FF_years = np.ones(shape=self.FF_frp.shape, dtype=int) * self.current_year
        self.FF_thresh_cases_per_yr = [fIn.getncattr('FF_thresh_cases_per_yr')]
        self.FF_thresh_yrs = [fIn.getncattr('FF_thresh_yrs')]
        try: self.FF_criteria = fIn.getncattr('FF_criteria')
        except: pass
        self.gridFF, self.gridNameFF = silamfile.read_grid_from_nc(fIn)
        fIn.close()
        # A map with cells marked as FF
        self.maskFF = np.zeros(shape=(self.gridFF.ny, self.gridFF.nx), dtype=np.byte)
        self.maskFF[self.FF_iy, self.FF_ix] = 1   # the actual mask 
        # A list of FF cells
        self.FF_indices = [[(np.zeros(shape=self.FF_frp.shape), self.FF_lat, self.FF_lon)]]
        self.ifNetcdf = True

    
    #===================================================================
    
    def read_namelist(self,frequent_fires_namelist_FNm):
        #
        # Reads a given namelist file with frequent fires
        #
        nlGrp = namelist.NamelistGroup.fromfile(frequent_fires_namelist_FNm)
        nlMeta = nlGrp.get('metadata')
        nlData = nlGrp.get('frequent_fires')
        self.FF_thresh_cases_per_yr = int(nlMeta.get_uniq('annual_nFires_threshold')) 
        self.FF_thresh_yrs = int(nlMeta.get_uniq('nYears_consecutive_threshold'))
        self.gridFF = gridtools.fromCDOnamelist(nlMeta)
        # Storing the whole multi-annual mask is too much, so shall keep just one year
        # as a mask and refresh it in case some other year is needed
        # prepare to store the list of lons, lats, and years of the mask
        ixFF = []
        iyFF = []
        yearsFF = []
        self.maskFF = np.zeros(shape=(self.gridFF.ny, self.gridFF.nx), dtype=np.byte)
        for item in nlData.get('frequent_fire'):
            flds = item.split()
            ixFF.append(int(flds[1]))
            iyFF.append(int(flds[2]))
            yearsFF.append(int(flds[3]))
        self.FF_ix = np.array(ixFF)
        self.FF_iy = np.array(iyFF)
        self.FF_years = np.array(yearsFF)
        self.current_year = np.nan
        self.FF_lon, self.FF_lat = self.gridFF.grid_to_geo(self.FF_ix, self.FF_iy)
        self.FF_indices = [[(np.zeros(shape=self.FF_ix.shape), self.FF_lat, self.FF_lon)]]
        self.ifNamelist = True


    #===================================================================
    
    def get_year_range(self, years2process):
        #
        # Calculates the start and end of counting for all required years and thresholds
        #
        self.idxYrStart = np.zeros(shape=(len(self.FF_thresh_yrs),len(years2process)),dtype=int)
        self.idxYrEnd = np.zeros(shape=(len(self.FF_thresh_yrs),len(years2process)),dtype=int)
        for iNYears, nYears in enumerate(self.FF_thresh_yrs):
            if len(years2process) < nYears:
                raise ValueError('Fewer years than the min_years threshold')
            # If many years are needed, each valid year will require a range of years to check
            self.idxYrStart[iNYears,:] = np.array(list((min(max(0, int(np.ceil(iYr-0.5*self.FF_thresh_yrs[iNYears]))),
                                                            len(years2process)-self.FF_thresh_yrs[iNYears]) 
                                                        for iYr in range(len(years2process)))))
            self.idxYrEnd[iNYears,:] = self.idxYrStart[iNYears,:] + nYears
            for iYr, yr in enumerate(years2process):
                self.log.log('Inclusive range for nYears=%i, year=%i, %s - %s' % 
                             (nYears, yr, years2process[self.idxYrStart[iNYears,iYr]],
                              years2process[self.idxYrEnd[iNYears,iYr]-1]))
        

    #===================================================================

    def massive_burning_regions(self):
        #
        # There are regions, which burn so massively that only very long time series
        # can reveal that these are the actual fires. Short-term (a year a two) analysis
        # marks them as frequent_fires and excludes.
        # Here we explicitly forbid exclusion of such regions if the time series are 
        # too short
        #
        mbAmazon = (np.array([-63.,-17.]),np.array([-49.,-7]), np.array([1]))   # ([lon_start, lat_start], [lon_end, lat_end], nYears)
        mbKalimantan = (np.array([112.,116.]),np.array([-4.,0.]), np.array([1,2,3,4,5]))   # ([lon_start, lat_start], [lon_end, lat_end], nYears)
        mbSumatra = (np.array([99.,106.]),np.array([-4.,0.5]), np.array([1,2,3,4,5]))   # ([lon_start, lat_start], [lon_end, lat_end], nYears)
        return [mbAmazon, mbKalimantan, mbSumatra]
        

    #===================================================================

    def find_frequent_fires_from_daily_maps(self, fire_src_templ, FreqFire_grid_FNm, 
                                            tStart, tEnd,
                                            FF_thresh_cases_per_yr, FF_thresh_yrs, dirOut):
        #
        # Scans the whole given period and sums-up the fires and FRP for a 3-km global grid
        # Builds a frequency histogram for the dataset.
        # Those grid cells, which have more than XX fires per year for over YY years will be 
        # masked out. The list of their locations will be the output.
        #
        self.FF_thresh_cases_per_yr = np.array(FF_thresh_cases_per_yr)  # list
        self.FF_thresh_yrs = np.array(FF_thresh_yrs)                    # list
        MapWrk = MxD_processor.daily_maps(log = self.log)
        years2process = range(tStart.year, tEnd.year + 1)
        self.get_year_range()
        # grid to work in
        self.gridFF = gridtools.fromCDOgrid(FreqFire_grid_FNm)
        QA_days = QA_problematic_days() 
        #
        # The main dataset: either read the map if previously stored or count fires.
        #
        self.mapFF_FRP = np.zeros(shape=(len(years2process), self.gridFF.ny, self.gridFF.nx))
        self.mapFF_nFires = np.zeros(shape=(len(years2process), self.gridFF.ny, self.gridFF.nx), 
                                     dtype=int)
        day = tStart
        while day <= tEnd:
            if QA_days.if_good_day(day, 'MOD') or QA_days.if_good_day(day, 'MYD'):
                if os.path.exists(day.strftime(fire_src_templ)):
                    print(day)
                else:
                    print('######## MISSING DAY ############', day) 
                    day += spp.one_day
                    continue
            else:
                print('######## BAD DAY ############', day) 
                day += spp.one_day
                continue
            MapWrk.from_file(day.strftime(fire_src_templ))
            # Add LU fires: count the occasions and sum-up FRP
            if MapWrk.nFires > 0:
                fx, fy = self.gridFF.geo_to_grid(MapWrk.FP_lon, MapWrk.FP_lat)
                ix = np.mod(np.round(fx).astype(int), self.mapFF_FRP.shape[2])
                iy = np.round(fy).astype(int)
                iYr = day.year-tStart.year
                try:
                    # Explicit cycle: ix and iy are not unique! Adding to a slice leads
                    # to a loss of multiple detections in a single grid cell
                    for iIdx in range(len(ix)):
                        self.mapFF_FRP[iYr, iy[iIdx], ix[iIdx]] += MapWrk.FP_frp[iIdx]
                        self.mapFF_nFires[iYr, iy[iIdx], ix[iIdx]] += 1
#                    self.mapFF_FRP[day.year-tStart.year, iy, ix] += MapWrk.FP_frp
#                    self.mapFF_nFires[day.year-tStart.year, iy, ix] += 1
#                    print('nFires tot, min, max, non-zero:',
#                          np.sum(self.mapFF_nFires[iYr,:,:]), 
#                          np.min(self.mapFF_nFires[iYr,:,:][self.mapFF_nFires[iYr,:,:]>0]),
#                          np.max(self.mapFF_nFires[iYr,:,:]),
#                          np.sum(self.mapFF_nFires[iYr,:,:]>0))
                except:
                    for i in range(len(ix)):
                        print('>>>>>>>> problem with grid indices ', MapWrk.FP_lon[i], 
                              MapWrk.FP_lat[i], fx[i], fy[i], ix[i], iy[i], day.year)
            day += spp.one_day
        #
        # Store the intermediate fire record file
        #
        spp.ensure_directory_MPI(dirOut)
        outF = silamfile.open_ncF_out(os.path.join(dirOut,'fire_map_%i-%i.nc4' % 
                                                   (years2process[0], years2process[-1])),
                                      'NETCDF4', self.gridFF, silamfile.SilamSurfaceVertical(), 
                                      tStart, list((dt.datetime(year,6,30) for year in years2process)), 
                                      [], 
                                      ['nFires_tot', 'FRP_raw',], 
                                      {'FRP_raw':'MW', 'nFires_tot':''},
                                      -999., True, 4, None)
        # grid of the WorkMap
        silamfile.write_grid_to_nc(outF, self.gridFF, 'grid_fire_records')
        # Now the variables
        for iYr, yr in enumerate(years2process):
            print('FRP, nFires for ', yr, np.sum(self.mapFF_FRP[iYr,:,:]), np.sum(self.mapFF_nFires[iYr,:,:]))
            outF['FRP_raw'][iYr,:,:] = self.mapFF_FRP[iYr,:,:]
            outF['nFires_tot'][iYr,:,:] = self.mapFF_nFires[iYr,:,:]
        outF.close()
        #
        # Having the map for all years, find out the cells with too many fires
        #
        self.detect_frequent_fires()
        #
        # Write them down to all formats we have
        #
        spp.ensure_directory_MPI(dirOut)
        self.frequent_fires_to_text_file(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.lst')
        self.frequent_fires_to_nc(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.nc4')
        self.frequent_fires_to_kml(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.kml')


    #============================================================================
    
    def find_frequent_fires_from_fire_records(self, fire_records_FNm, tStart, tEnd,
                                              FF_thresh_cases_per_yr, FF_thresh_yrs, dirOut):
        #
        # A high-res global map with fire records each year is a nice intermediate
        # if new thresholds are needed
        #
        self.FF_thresh_cases_per_yr = np.array(FF_thresh_cases_per_yr)
        self.FF_thresh_yrs = np.array(FF_thresh_yrs)
        years2process = range(tStart.year, tEnd.year + 1)
        self.get_year_range(years2process)
        print('Reading fire records...', fire_records_FNm)
        inF = nc4.Dataset(os.path.join(dirOut,fire_records_FNm),'r')
        inF.set_auto_mask(False)
        self.mapFF_FRP = inF.variables['FRP_raw'][:,:,:]
        self.mapFF_nFires = inF.variables['nFires_tot'][:,:,:]
        self.gridFF, self.gridNameFF = silamfile.read_grid_from_nc(inF)
        inF.close()
        #
        # Having the map for all years, find out the cells with too many fires
        #
        self.detect_frequent_fires(years2process)
        #
        # Write them down to all formats we have
        #
        print('Storing output for ', years2process)
        spp.ensure_directory_MPI(dirOut)
        self.frequent_fires_to_text_file(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.lst')
        self.frequent_fires_to_nc(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.nc4')
        self.frequent_fires_to_kml(dirOut, years2process, 'Frequent_fires_ann_%i_nYrs_%i_%i.kml')
        

    #====================================================================
    
    def detect_frequent_fires(self, years2process):
        #
        # Scans the fire record map and creates a list of frequent fires
        # argwhere returns a list of 3-element vectors ([year, lat, lon])
        #
        self.FF_indices = []

        for iNYears, nYears in enumerate(self.FF_thresh_yrs):
            self.FF_indices.append([])
            #
            # Massively burning regions are allowed to have many fires within one year
            # 
            mapMassiveBurn = np.ones(shape=self.mapFF_nFires.shape)
            for mbCoord in self.massive_burning_regions():
                if nYears in mbCoord[2] or mbCoord[2][0] == 0:
                    xSE, ySE = self.gridFF.geo_to_grid(mbCoord[0],mbCoord[1])
                    mapMassiveBurn[:,int(ySE[0]):int(ySE[1]),int(xSE[0]):int(xSE[1])] = 0.

            for nFires in self.FF_thresh_cases_per_yr:
                #
                # First, find all places and years with many fires
                #
                mapTmp = (self.mapFF_nFires > nFires) * mapMassiveBurn
                mapTmp2 = np.zeros(shape=mapTmp.shape)
                #
                # For each year, get the places that satisfy the multi-annual requirement,
                # i.e. all years within the range to count have excess fires over the threshold
                #
                for iYr in range(len(years2process)):
                    mapTmp2[iYr,:,:] = np.product(mapTmp[self.idxYrStart[iNYears, iYr] : 
                                                         self.idxYrEnd[iNYears, iYr],
                                                         :,:], axis=0)

                self.FF_indices[-1].append(np.argwhere(mapTmp2 > 0))


    #====================================================================

    def add(self, FFnew):
        #
        # Merges two criteria summing-up the frequent fires masks
        #
        if self.gridFF is None:
            self.gridFF = copy.deepcopy(FFnew.gridFF)
            self.FF_frp = copy.deepcopy(FFnew.FF_frp)
            self.FF_ix = copy.deepcopy(FFnew.FF_ix)
            self.FF_iy = copy.deepcopy(FFnew.FF_iy)
            self.FF_lat = copy.deepcopy(FFnew.FF_lat)
            self.FF_lon = copy.deepcopy(FFnew.FF_lon)
            self.FF_nFires = copy.deepcopy(FFnew.FF_nFires)
            self.FF_thresh_cases_per_yr = copy.deepcopy(FFnew.FF_thresh_cases_per_yr)
            self.FF_thresh_yrs = copy.deepcopy(FFnew.FF_thresh_yrs)
            self.FF_years = copy.deepcopy(FFnew.FF_years)
            self.current_year = FFnew.current_year
        else:
            if self.gridFF == FFnew.gridFF and self.current_year == FFnew.current_year:
                self.log.log('Summing-up frequent fires')
            else:
                self.log.log('Grids or years of two FF objects differ')
                raise ValueError('Grids or years of two FF objects differ')
            self.FF_frp = np.concatenate([self.FF_frp, FFnew.FF_frp])
            self.FF_ix = np.concatenate([self.FF_ix, FFnew.FF_ix])
            self.FF_iy = np.concatenate([self.FF_iy, FFnew.FF_iy])
            self.FF_lat = np.concatenate([self.FF_lat, FFnew.FF_lat])
            self.FF_lon = np.concatenate([self.FF_lon, FFnew.FF_lon])
            self.FF_nFires = np.concatenate([self.FF_nFires, FFnew.FF_nFires])
            self.FF_years = np.concatenate([self.FF_years, FFnew.FF_years])
            self.FF_criteria = '_'.join(list(('nF%i_nY%i' % (FF.FF_thresh_cases_per_yr[0],
                                                             FF.FF_thresh_yrs[0])
                                              for FF in [self, FFnew])))
            self.FF_thresh_cases_per_yr = [0]
            self.FF_thresh_yrs = [0]
        #
        # The FF_indices are for just one criterion, which is a combination of several
        # 
        self.FF_indices = [[np.zeros(shape=(self.FF_frp.shape[0],3),dtype=np.int64)]]
#        self.FF_indices[0][0][:,0] = self.FF_years[:]
        self.FF_indices[0][0][:,1] = self.FF_iy[:]
        self.FF_indices[0][0][:,2] = self.FF_ix[:]
        

#                                     np.ones(shape=self.FF_frp.shape) * self.current_year,
#                             self.FF_lat, self.FF_lon)]]


    #====================================================================
    
    def frequent_fires_to_text_file(self, outDir, years2process, chFNmOut):
        #
        # Store to the text file
        #
        for iNYears, nYears in enumerate(self.FF_thresh_yrs):
            for iNFires, nFires in enumerate(self.FF_thresh_cases_per_yr):
                FFidx = self.FF_indices[iNYears][iNFires]
                for iYr, yr in enumerate(list(set(FFidx[:,0]))):
                    print('iNYears, nYears: ', iNYears, nYears, 'iNFires, nFires: ',iNFires, nFires, 'iYr, yr:',iYr, yr)
                    idxYr = FFidx[:,0] == yr
                    lons_FF, lats_FF = self.gridFF.grid_to_geo(FFidx[idxYr,2], FFidx[idxYr,1])
                    iy = FFidx[idxYr,1]
                    ix = FFidx[idxYr,2]
                    if nFires == 0 and nYears == 0:
                        outF = open(os.path.join(outDir,chFNmOut % (self.FF_criteria, years2process[yr])), 'w')
                        outF.write('LIST = metadata\n')
                        outF.write('multicriteria = %s\n' % self.FF_criteria)
                    else:
                        outF = open(os.path.join(outDir,chFNmOut % (nFires, nYears, years2process[yr])), 'w')
                        outF.write('LIST = metadata\n')
                        outF.write('annual_nFires_threshold = %i\n' % nFires)
                        outF.write('nYears_consecutive_threshold = %i\n' % nYears)
                        outF.write('counted_years = %i - %i\n' % (self.idxYrStart[iNYears, iYr],
                                                                  self.idxYrEnd[iNYears, iYr]))
                    for s in self.gridFF.toCDOgrid().tostr(): 
                        outF.write(s)
                    outF.write('\nEND_LIST = metadata\n')
                    outF.write('## frequent_fire = <nbr> <ix> <iy> <year> <FRP_nYrs> <nFires_nYrs> <lon> <lat>\n')
                    outF.write('LIST = frequent_fires\n')
                    if nFires == 0 and nYears == 0:
                        for iFF in range(np.sum(idxYr)):  # lines for this year
                            outF.write('frequent_fire = %i %i %i %i %g %g %g %g\n' % 
                                       (iFF, ix[iFF], iy[iFF], years2process[yr],
                                        self.FF_frp[iFF], self.FF_nFires[iFF], 
                                        lons_FF[iFF], lats_FF[iFF]))
                    else:
                        for iFF in range(np.sum(idxYr)):  # lines for this year
                            outF.write('frequent_fire = %i %i %i %i %g %g %g %g\n' % 
                                       (iFF, ix[iFF], iy[iFF], years2process[yr], 
                                        np.sum(self.mapFF_FRP[self.idxYrStart[iNYears, iYr] : 
                                                              self.idxYrEnd[iNYears, iYr],
                                                              iy[iFF], ix[iFF]]), 
                                        np.sum(self.mapFF_nFires[self.idxYrStart[iNYears, iYr] : 
                                                                 self.idxYrStart[iNYears, iYr],
                                                                 iy[iFF], ix[iFF]]),
                                        lons_FF[iFF], lats_FF[iFF]))
                    outF.write('END_LIST = frequent_fires\n')
                    outF.close()


    #=======================================================================================

    def frequent_fires_to_nc_data(self, dirOut, years2process, chFNmOut):
        #
        # Store the frequent fires to netcdf
        #
        for iNYears, nYears in enumerate(self.FF_thresh_yrs):
            for iNFires, nFires in enumerate(self.FF_thresh_cases_per_yr):
                FFidx = self.FF_indices[iNYears][iNFires]
                for iYr, yr in enumerate(list(set(FFidx[:,0]))):
                    idxYr = FFidx[:,0] == yr
                    lons_FF, lats_FF = self.gridFF.grid_to_geo(FFidx[idxYr,2], FFidx[idxYr,1])
                    self.iy = FFidx[idxYr,1]
                    self.ix = FFidx[idxYr,2]
                    #
                    # Panoply is stupid enough to break down trying to draw vectors if lons or lats 
                    # have exactly same values. Their sorter requires strict relation.
                    # Let's add a random value about 1 m to avoid problems
                    #
                    self.lons += (np.random.rand(len(lons_FF)) - 0.5) / 100000.
                    self.lats += (np.random.rand(len(lons_FF)) - 0.5) / 100000.
                    if nFires == 0 and nYears == 0:
                        chFNm = os.path.join(dirOut,chFNmOut % (self.FF_criteria, years2process[yr]))
                    else:
                        chFNm = os.path.join(dirOut, chFNmOut % (nFires,nYears,years2process[yr]))

                    # variables
                    # If mapFF_FRP and .._nFires exist, use them to create the vectors
                    # If vectors are available already, just use them directly (map is then undefined)
                    try:
                        self.frp = np.sum(self.mapFF_FRP[self.idxYrStart[iNYears,iYr] : 
                                                         self.idxYrEnd[iNYears,iYr],
                                                         self.QA_basic.iy[:],
                                                         self.QA_basic.ix[:]], axis=0)
                        self.nFires = np.sum(self.mapFF_nFires[self.idxYrStart[iNYears,iYr] : 
                                                               self.idxYrEnd[iNYears,iYr],
                                                               self.QA_basic.iy[:],
                                                               self.QA_basic.ix[:]], axis=0)
                    except: pass
                    #
                    # Write the stuff to the file
                    #
                    self.to_nc(chFNm, {'FF_year' : years2process[yr], 'FF_thresh_cases_per_yr' : nFires,
                                       'FF_thresh_yrs' : nYears, 'FF_criteria' : self.FF_criteria,
                                       'FF_first_counted_year' : self.idxYrStart[iNYears,iYr],
                                       'FF_last_counted_year' : self.idxYrEnd[iNYears,iYr]})
        

    #====================================================================
    
    def frequent_fires_to_kml(self, dirOut, years2process, chFNmOut):  #, chStyle):
        #
        # Stores the list of frequent fires into a list of "My places" for GoogleEarth
        #
        for iNYears, nYears in enumerate(self.FF_thresh_yrs):
            for iNFires, nFires in enumerate(self.FF_thresh_cases_per_yr):
                FFidx = self.FF_indices[iNYears][iNFires]
                for iYr, yr in enumerate(list(set(FFidx[:,0]))):
                    idxYr = FFidx[:,0] == yr
                    if not np.any(idxYr): continue
                    iy = FFidx[idxYr,1]
                    ix = FFidx[idxYr,2]
                    if nFires == 0 and nYears == 0:
                        chFNm = os.path.join(dirOut, chFNmOut % (self.FF_criteria,years2process[yr]))
                        chName = self.FF_criteria
                        self.QA_basic.lons = self.FF_lon
                        self.QA_basic.lats = self.FF_lat
                        self.QA_basic.nFires = self.FF_nFires
                    else:
                        chFNm = os.path.join(dirOut, chFNmOut % (nFires,nYears,years2process[yr]))
                        chName = 'nPerYr=%g nYr=%g' % (nFires, nYears)
                        self.lons, self.lats = self.gridFF.grid_to_geo(FFidx[idxYr,2], FFidx[idxYr,1])
                        self.nFires = np.sum(self.mapFF_nFires[self.idxYrStart[iNYears,iYr] : self.idxYrEnd[iNYears,iYr],
                                                                        iy, ix], axis=0)
                    # Now, use the basic function
                    #
                    self.to_kml(chFNm, chName, 'FF', str(years2process[yr]))
                    

    #====================================================================
    
    def draw_FF_histogram(self, outDir):
        #
        # Draws basic information on the frequent fires
        #
        histFRP, binsFRP  = np.histogram(self.mapFF_FRP, 50)
        histNFires, binsNFires = np.histogram(self.mapFF_nFires, 50)
        #
        # Draw histograms
        #
        fig, axes = plt.subplots(1,3, figsize=(15,9))
    #    axes[0].plot(binsNFires[2:], histNFires[1:], label='Nbr of fires per year')
        axes[0].semilogy(binsNFires[1:], histNFires[:], label='Nbr of fires per year')
        axes[0].set_xlabel('Nbr of fire records in a year')
        axes[0].set_ylabel('Nbr of grid cells')
        axes[0].set_title('Histogram of Nbr of fire registrations')
        axes[0].grid(True)
    #    ax2=axes[0].twinx()
    #    ax2.plot(binsNFires[2:],np.cumsum(histNFires)[1:] / np.cumsum(histNFires)[-1], c='orange')
    #    ax2.set_ylabel('Fraction of grid cells',color='orange')
    #    ax2.set_yticklabel(color='orange')
    #    ax2.ticklabel_format(useOffset=False,axis='y')
        
    #    axes[1].plot(binsFRP[2:], histFRP[1:], label = 'FRP sum')
        axes[1].semilogy(binsFRP[1:], histFRP[:], label = 'FRP sum')
        axes[1].set_xlabel('Sum FRP, MW')
        axes[1].set_ylabel('Nbr of grid cells')
        axes[1].set_title('Histogram of total FRP release')
        axes[1].grid(True)
    #    ax3=axes[1].twinx()
    #    ax3.plot(binsFRP[2:],np.cumsum(histFRP)[1:] / np.cumsum(histFRP)[-1], c='orange')
    #    ax3.set_ylabel('Fraction of FRP',color='orange')
    #    ax3.set_yticklabel(color='orange')
    #    ax3.ticklabel_format(useOffset=False,axis='y')
        
        axes[2].scatter(self.mapFF_nFires[self.mapFF_FRP>0], 
                        self.mapFF_FRP[self.mapFF_FRP>0], label='FRP vs nbrof cases',
                        marker='.')
        axes[2].set_xlabel('Number of fires')
        axes[2].set_ylabel('FRP')
        axes[2].set_title('FRP vs number of fires registered in a year')
        axes[2].grid(True)
        
        plt.savefig(os.path.join(outDir, 'histogr_FRP_NFires.png'), dpi=300, layout='tight')
        plt.clf()
        plt.close()


    #======================================================================

    def draw_annual_FF_map(self, iNFires, iNYears, iYr, dirOut):
        #
        # Reads the file and draws two maps in one page: total FRP and number of fires
        # for the given annual frequent-fire nc file  
        #
        if len(self.FF_lon) == 0:
            self.log.log('No fires for nFiresPerYear %i, nYears %i, current_year %i' %
                         (self.FF_thresh_cases_per_yr, self.FF_thresh_yrs, self.current_year))
            return
        spp.ensure_directory_MPI(os.path.join(dirOut,'FRP'))
        spp.ensure_directory_MPI(os.path.join(dirOut,'nFires'))

        if self.FF_thresh_cases_per_yr[0] == 0:
            chTitle = 'Freq. fires, %s, ' + '%s, %i' % (self.FF_criteria,self.current_year)
            chFNmOut = 'FF_%s_' + '%s_%i.png' % (self.FF_criteria, self.current_year)
        else:
            chTitle = 'Freq.fires, %s,' +', nFFperYrMin=%i, nYrsMin=%i, %i' % (nFires, nYears, yr)
            chFNmOut = 'FF_%s_' + 'ann_%i_nYrs_%i_%i.png' % (self.FF_thresh_cases_per_yr[0],
                                                             self.FF_thresh_yrs[0], self.current_year)
        drawer.draw_map_with_points(chTitle % 'FRP',
                                    self.FF_lon, self.FF_lat, self.gridFF, 
                                    os.path.join(dirOut,'FRP'), chFNmOut % 'FRP', 
                                    vals_points=self.FF_frp, vals_map=None, chUnit='MW', 
                                    numrange=(np.nan,np.nan), ifNegatives=False, zipOut=None,
                                    cmap='rainbow')
        drawer.draw_map_with_points(chTitle % 'nFires',
                                    self.FF_lon, self.FF_lat, self.gridFF, 
                                    os.path.join(dirOut,'nFires'), chFNmOut % 'nFires', 
                                    vals_points=self.FF_nFires, vals_map=None, chUnit='nbr', 
                                    numrange=(np.nan,np.nan), ifNegatives=False, zipOut=None, 
                                    cmap='rainbow')
        

    #======================================================================

    def get_mask(self, lons, lats, day):
        #
        # In the list of longitudes and latitudes finds those that correspond to
        # frequent fires grid cells and returns the corresponding mask
        #
        # The mask may be underfined if we are at the first step - or another year is needed
        if self.current_year != day.year: 
            self.update_mask_4day(day)
        #
        # turn input geo to FF map indices
        #
        fx, fy = self.gridFF.geo_to_grid(lons, lats)
        ix = np.mod(np.round(fx).astype(int), self.gridFF.nx)  # modulo needed to close the globe
        iy = np.round(fy).astype(int)
        # return the mask
        return self.maskFF[iy, ix] == 1  # Zero means good (not frequent fire)


    #======================================================================

    def update_mask_4day(self, day):
        # What do we have / need?
        # Does the grid exist?
        if self.gridFF is None: 
            try:
                self.read_nc(day.strftime(self.FNm_template))  # sets all
                self.log.log('QA_FF updated from netcdf to ' + str(day.year))
            except:
                self.read_namelist(day.strftime(self.FNm_template))
                self.log.log('QA_FF updated from namelist to ' + str(day.year))
            idx = self.FF_years == day.year    # namelist can be made for many years
            self.maskFF = np.zeros(shape=(self.gridFF.ny, self.gridFF.nx), dtype=np.byte)
            self.maskFF[self.FF_iy[idx], self.FF_ix[idx]] = 1  
            self.current_year = day.year
        #
        # Grid is defined but year may be wrong
        #
        elif self.current_year != day.year:
            if self.ifNetcdf:
                try:
                    self.read_nc(day.strftime(self.FNm_template))
                    self.log.log('QA_FF updated from netcdf to ' + str(day.year))
                except: # no netcdf file but grid is defined, so this is a missing year. EXTRAPOLATE
                    self.log.log('QA extrapolates %i to %i' % (self.current_year, day.year))
                    self.current_year = day.year
            elif self.ifNamelist:
                idx = self.FF_years == day.year    # namelist can be made for many years
                self.maskFF[self.FF_iy[idx], self.FF_ix[idx]] = 1  
                self.current_year = day.year
            else: raise ValueError('Neither nc nor namelist input is available')
        else:
            raise ValueError('QA get_mask does not understand, why called')
        

    #======================================================================

    def convert_5wFF_to_FF(self, year, chInFNm, dirOut):
        #
        # Converts a 5w-RK format to FF object and writes it down as FF-netcdf
        #
        self.FF_thresh_cases_per_yr = -5  # that will mean 5 weeks
        self.FF_thresh_yrs = 1
        self.gridFF = None
        self.idxYrStart = [0]
        self.idxYrEnd = [1]
        #
        # Get the 5w map
        #
        rdrMask = silamfile.SilamNCFile(chInFNm).get_reader('mask', mask_mode=False)
        #
        # The first year decides the dimensions and creates the array 
        #
        if self.gridFF is None:
            self.gridFF = gridtools.latLonFromPoints(rdrMask.coordinates()[1][0,:],  # lats
                                                     rdrMask.coordinates()[0][:,0])  # lons
            self.mapFF_FRP = np.zeros(shape=(1, self.gridFF.ny, self.gridFF.nx), dtype=np.float32)
            self.mapFF_nFires = np.zeros(shape=(1, self.gridFF.ny, self.gridFF.nx), dtype=int)
        #
        # The main dataset: either read the map if previously stored or count fires.
        #
        self.mapFF_FRP[0] = rdrMask.read(1).T
        self.mapFF_nFires = self.mapFF_FRP   # since 5w mask is just a flag, no extra info is available
        idxTmp = np.where(self.mapFF_nFires > 0)
        self.FF_indices = np.zeros(shape=(len(idxTmp[0]),3),dtype=int)
        for i in range(3): self.FF_indices[:,i] = idxTmp[i][:]
        self.FF_years = np.ones(shape=(len(idxTmp[0]))) * year
        #
        # Now store it to the netcdf file
        #
        spp.ensure_directory_MPI(dirOut)
        self.frequent_fires_to_nc(dirOut, 0,
                                  'FreqFires_from_5w_%s.nc4'.replace('.nc','') 
                                  % os.path.split(chInFNm)[-1])
        self.frequent_fires_to_kml(dirOut, 0, 'FreqFires_from_5w_%s.kml'.replace('.nc','') 
                                   % os.path.split(chInFNm)[-1])

    #===========================================================================

    def report(self):
        #
        try:
            return 'Frequent fires %s annual_nFires = %i nYears_consecutive = %i counted_years = %i - %i %s' % (
                self.FF_criteria, nFires, nYears, self.idxYrStart[iNYears, iYr], self.idxYrEnd[iNYears, iYr],
                self.gridFF.toCDOgrid().tostr())
        except: return 'Frequent fires: No meaningful QA'




#############################################################################

class QA_sml_huge_fires():
    #
    # Removes fires with FRP larger than the prescribed value
    
    #======================================================================

    def __init__(self, minFRP_MW, maxFRP_MW, log):
        self.minFRP_MW = minFRP_MW
        self.maxFRP_MW = maxFRP_MW
        if self.minFRP_MW is None or self.maxFRP_MW is None: self.QA_flag = np.int64(0) # no QA
        else: self.QA_flag = def_sml_huge_fires_flag  # active QA
        self.log = log

    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni):
        #
        # In the IS4FIRES v.3.0 control file huge fires are represented
        # via the max FRP allowed
        #
        return cls(float(nlIni.get_uniq('min_max_fire_power_MW').split()[0]),
                   float(nlIni.get_uniq('min_max_fire_power_MW').split()[1]),
                   nlIni.get_uniq('QA_log'))


    #===============================================================
    @classmethod
    def void(cls):
        #
        QA_flag = np.int64(0)
        return cls(None, None, None)

    #===============================================================
    @classmethod
    def from_nc(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        # Retrieve parameters and initiate
        try: 
            return cls(fIn.QA_minFRP_MW, fIn.QA_maxFRP_MW, log)
        except: return cls.void()

    #===============================================================
    
    def to_nc(self, fOut):
        #
        if self.QA_flag: 
            fOut.setncattr('QA_minFRP_MW',self.minFRP_MW)
            fOut.setncattr('QA_maxFRP_MW',self.maxFRP_MW)

    #======================================================================

    def get_mask(self, FP_frp):
        return np.logical_or(FP_frp < self.minFRP_MW, FP_frp > self.maxFRP_MW)

    #===========================================================================

    def report(self):
        #
        if self.minFRP_MW is None or self.maxFRP_MW is None: return 'Small_huge_fires QA is OFF'
        return 'Small - huge fires threshold %f - %f' % (self.minFRP_MW, self.maxFRP_MW)



#############################################################################

class QA_void_fires():
    #
    # Some fires are reported over water or otherwise unrecognised land-use
    # They might be useful for something, e.g. marine oil platform handling
    # but not for vegetation fires. Here we remove them

    def __init__(self, chVoidLabel, all_LUs, log):
        self.chVoidLabel = chVoidLabel
        self.all_LUs = copy.deepcopy(all_LUs)
        self.lstLU_void = []
        if chVoidLabel == '': 
            self.QA_flag = np.int64(0)  # No QA
            return  # no QA
        for i, lu in enumerate(all_LUs):
            if self.chVoidLabel in lu: self.lstLU_void.append(i)
        self.QA_flag = def_void_fires_flag  # Active QA
        self.log = log

    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni, LUs):
        #
        # Bad LU needs to be removed. Which one?
        #
        return cls(nlIni.get_uniq('void_land_use'), LUs, nlIni.get_uniq('QA_log'))

    #===============================================================
    @classmethod
    def void(cls):
        #
        # Bad LU needs to be removed. Which one?
        #
        return cls('', [], None)

    #===============================================================
    @classmethod
    def from_nc(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        # Retrieve parameters and initiate
        try: 
            return cls(fIn.QA_void_land_use, fIn.QA_all_LUs.split(), log)  # name of void LU and list of all LUs
        except: return cls.void()

    #===============================================================
    
    def to_nc(self, fOut):
        #
        if self.QA_flag: 
            fOut.setncattr('QA_void_land_use',self.chVoidLabel)
            fOut.setncattr('QA_all_LUs',' '.join(self.all_LUs))

    #===========================================================================

    def get_mask(self, FP_LU):
        if len(self.lstLU_void) == 0: return np.ones(shape=(len(FP_LU)),dtype=bool)
        maskBad = FP_LU == self.lstLU_void[0]
        for luVoid in self.lstLU_void[1:]:
            maskBad = np.logical_or(maskBad, FP_LU == luVoid)
        return maskBad

    #===========================================================================

    def report(self):
        #
        if len(self.lstLU_void) == 0:
            return 'No void LU types; '
        elif len(self.lstLU_void) == 1:
            return 'Void LU type %g' % self.lstLU_void[0]
        else:
            return 'Void LU %s' % ' '.join(str(self.lstLU_void))


############################################################################

class QA_problematic_days():
    #
    # Some days have enourmous amount of fires in strange places.
    # Current solution is with dictionary but with so small sets it should be fine
    #
    def __init__(self, sources_avail, log):
        if sources_avail is None:
            self.QA_flag = np.int64(0)  # no QA expected
            self.sources = None
            return
        else:
            if type(sources_avail) == str: self.sources = [sources_avail]
            else: self.sources = sources_avail
        # nothing before...
        self.start_day = {'MOD':dt.date(2000,3,1),
                          b'T':dt.date(2000,3,1),
                          'MYD':dt.date(2002,7,4),
                          b'A':dt.date(2002,7,4),
                          'MxD':dt.date(2002,7,4),  # both MOD/MYD must exist
                          'VJ1':dt.date(2018,1,5),
                          b'1':dt.date(2018,1,5),
                          'VNP':dt.date(2012,1,19),
                          b'N':dt.date(2012,1,19),
                          'Vxx':dt.date(2018,1,5), # both VNP and VJ1 must exist
                          '-':dt.date(2000,3,1)}   # unknown satellite => do nothing
        # This-far, bad days were for Terra at the beginning
        self.bad_days = {'MOD': [dt.date(2000,2,24),
                                 dt.date(2000,2,25), 
                                 dt.date(2000,2,26), 
                                 dt.date(2000,2,28),
                                 dt.date(2000,3,5),
                                 dt.date(2000,3,15),
                                 dt.date(2000,10,30),
                                 dt.date(2000,10,31)],
                         b'T': [dt.date(2000,2,24),
                                dt.date(2000,2,25), 
                                dt.date(2000,2,26), 
                                dt.date(2000,2,28),
                                dt.date(2000,3,5),
                                dt.date(2000,3,15),
                                dt.date(2000,10,30),
                                dt.date(2000,10,31)],
                         # Aqua is fine
                         'MYD': [],
                         b'A': [],
                         # MxD is problematic if any of MODIS has a problem
                         'MxD': [dt.date(2000,2,24),
                                 dt.date(2000,2,25), 
                                 dt.date(2000,2,26), 
                                 dt.date(2000,2,28),
                                 dt.date(2000,3,5),
                                 dt.date(2000,3,15),
                                 dt.date(2000,10,30),
                                 dt.date(2000,10,31)],                         # For VIIRS no info about problems, so far
                         'VJ1':[], b'1':[],
                         'VNP':[dt.date(2012,1,19)],
                         'Vxx':[dt.date(2012,1,19)],
                         b'N':[dt.date(2012,1,19)]}
        self.QA_flag = def_problematic_days_flag  # active QA expected
        self.log = log
        
    #===============================================================
    @classmethod
    def void(cls):
        # cancels all checks
        return cls(None, None)
    
    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni):
        #
        # In the IS4FIRES v.3.0 control file huge fires are represented
        # via the max FRP allowed
        #
        if nlIni.get_uniq('bad_days').upper() != 'NONE':     # REMOVE_WHOLE_DAY / REMOVE_BAD_SOURCE / NONE
            return cls(nlIni.get('satellite_to_check'), nlIni.get_uniq('QA_log'))
        else:
            return cls.void()

    #===============================================================
    @classmethod
    def from_nc(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        # Retrieve parameters and initiate
        try: 
            if fIn.QA_bad_days.upper() == 'REMOVE': 
                satellites = fIn.QA_satellites.split()
                return cls(satellites, log)
            else: return cls.void()
        except: return cls.void()

    #===============================================================
    
    def to_nc(self, fOut):
        #
        if self.QA_flag: 
            fOut.setncattr('QA_bad_days','REMOVE')
            fOut.setncattr('QA_satellites',' '.join(self.sources))


    #======================================================================

    def if_bad_day(self, day):
        
        for src in self.sources:
#            if day < self.start_day[src]: return True         # before the start => bad day. Not really, just there will be no fires anyway
            idxDay = np.searchsorted(self.bad_days[src], day.date())    # day position in the list of bad days
            if idxDay < len(self.bad_days[src]):
                if day.date() == self.bad_days[src][idxDay]: return True    # found in the list of bad days => bad day
        return False    # no alarms raised by any of the sources 
        self.QA_flag = np.int64(1)  # active QA expected


    #======================================================================

    def good_sources(self, day):
        outLst = copy.copy(self.sources)
        if self.sources is None: return outLst
        for src in list(set(self.bad_days.keys()).intersection(set(outLst))):
            if day < self.start_day[src]: outLst.remove(src)
            else:
                if np.searchsorted(self.bad_days[src], day) >= len(self.bad_days[src]): continue
                if self.bad_days[src][np.searchsorted(self.bad_days[src], day)] == day: 
                    outLst.remove(src)
        return outLst

    #======================================================================
    
    def get_mask(self, day):
        if self.sources is None: return False  # whatever, void check
        return self.if_bad_day(day)

    #===========================================================================

    def report(self):
        #
        return 'Bad days removal active'


#################################################################################

class QA_glints(QA_basic):
    #
    # Sun glints over water are flagged at the satellite level 1, but there may be
    # glints due to roof or a small lake. These are visible only at a specific relation
    # between the sun zenith angle, viewing angle, and viewing direction.
    # They can be caught if a fire is reported every time the satellite passes at the
    # same position, i.e., they must have a cycle equal to the revisit time of the satellite
    #
    def __init__(self, chMethod, glints_FNm_templ, chDirMetadata, log):
        super().__init__(log)
#        self.chGlintFNmTempl = chGlintFNmTempl  # must have a year in the name as a template
        if chMethod is None or chMethod == 'NONE':
            self.QA_flag = np.int64(0)  # nothing 
            self.log = log
            return
        self.method = chMethod
        if self.method == 'mask':
            self.glint_string = '# glint = i1d, ix, iy, lon, lat, nFires  # [dates_of_fires]'
        else: raise ValueError('Unknown glint method')
        if chDirMetadata is None: self.FNm_templ = glints_FNm_templ
        else: self.FNm_templ = os.path.join(chDirMetadata, glints_FNm_templ)
        self.today = None
        self.QA_flag = def_glints_flag  # active QA expected
        self.log = log

    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni, chDirMetadata, log):
        #
        # Read them
        return cls(nlIni.get_uniq('glints_method'), nlIni.get_uniq('glints_FNm_template'), chDirMetadata, log)

    #===============================================================
    @classmethod
    def from_nc_QA(cls, fIn_, log):
        #
        # Reads the generic description of the glint object from the QA general file
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        #
        # Retrieve parameters and initiate
        #
        try: 
            return cls(fIn.QA_glints_method, fIn.QA_glints_FNm_template, None, log)
        except: 
            return cls(None, None, None, None)  # no parameters stored, void QA_glints

    #===============================================================

    def from_nc_data(self, fIn_, log):
        #
        # Reads the glint data from the given file
        #
        # Need to open?
        if isinstance(fIn_, str): 
            fIn = nc4.Dataset(fIn_,'r')
            fIn.set_auto_maskandscale(False) ## Never mask, never scale
        else: fIn = fIn_
        #
        # grid
        #
        self.grid, self.gridName = silamfile.read_grid_from_nc(fIn)
        #
        # Glints themselves
        #
        self.frp = fIn.variables['frp'][:].astype(np.float32)
        self.lon = fIn.variables['lon'][:].astype(np.float32)
        self.lat = fIn.variables['lat'][:].astype(np.float32)
        self.nFires= fIn.variables['nfires'][:].astype(np.int16)
        self.ix = fIn.variables['ix'][:].astype(np.int32)
        self.iy = fIn.variables['iy'][:].astype(np.int32)

        self.i1d = self.ix + self.iy * self.grid.nx

        fIn.close()
        
        
    #===============================================================
    
    def to_nc_QA(self, fOut):
        #
        # Writes the specification of the glint objects, not the data themselves
        #
        if self.QA_flag: 
            fOut.setncattr('QA_glints_method', self.method)
            fOut.setncattr('QA_glints_FNm_template', self.FNm_templ)

    #===============================================================
    
    def to_nc_data(self, fOut_):
        #
        # Writes the internal glint data into the given file
        #
        # file name or handler?
        if isinstance(fOut_, str): 
            fOut = nc4.Dataset(fOut_, "w", format="NETCDF4")
        else: fOut = fOut_
        #
        # grid
        #
        silamfile.write_grid_to_nc(fOut, self.grid, self.gridName)
        fOut.grid_projection = 'lonlat'
        #
        # Dimension: fires
        #
        fires = fOut.createDimension("fires",len(self.nFires))  # nFires here is the number of "fire" observations of the particular glint
        
        for FP_var in [('frp','f4','FRP','MW', self.frp),   # var_name, type, long name, unit
                       ('lon','f4','longitude','degrees_east', self.lon),
                       ('lat','f4','latitude','degrees_north', self.lat),
                       ('nfires','i4','Nbr of fires','', self.nFires),
                       ('ix','i4','pixel x-index','', self.ix),
                       ('iy','i4','pixel y-index','', self.iy)]:
            vFP = fOut.createVariable(FP_var[0], FP_var[1], ("fires"), zlib=True, complevel=5)
            vFP.long_name = FP_var[2]
            if FP_var[3] != '': vFP.units = FP_var[3]
            fOut.variables[FP_var[0]][:] = FP_var[4][:]

        fOut.close()


    #=====================================================================

    def read_glints_txt(self, chFNmIn):
        #
        # Reads the glint file for the given year
        #
        nlGrp = namelist.NamelistGroup.fromfile(chFNmIn,'r')
        glints = nlGrp.get('glint').values()[0]
        nlParams = nlGrp.get('parameters')
        self.grid = gridtools.fromCDOnamelist(nlParams)
        self.gridNm = nlParams.get_uniq('grid_name')
        self.method = nlParams.get_uniq('method')
        self.years = nlParams.get_uniq('years')
        self.i1d = np.array(list(((np.int64(g.split()[0]) for g in glints))), dtype=np.int64)
        self.lons = np.array(list((np.float32(g.split()[3]) for g in glints)), dtype=np.float32)
        self.lats = np.array(list((np.float32(g.split()[4]) for g in glints)), dtype=np.float32)
        self.nFires = np.array(list((int(g.split()[5]) for g in glints)), dtype=int)
        self.frps = np.array(list((np.float32(g.split()[6]) for g in glints)), dtype=np.float32)
        if self.method == 'mask':
            self.chDates = np.array(list((np.float32(g.split()[7]) for g in glints)))
        else: raise ValueError('Unknown glint method')

    #=====================================================================
    
    def read_glints_nc(self, today):
        #
        # Reads glints for the given time.
        #
        fIn = nc4.Dataset(today, strftime(self.FNm_templ),'r')
        self.grid, self.gridName = silamfile.read_grid_from_nc(fIn)
        

    #=====================================================================

    def get_mask(self, lons, lats, today):
        #
        # Projects the given lons and lats to the grid of glints and returns a filter of glints
        #
        # Do we have the data for the right time? Note that the day itself may not matter
        # As long as the glint file name is the same, all is fine 
        #
        if self.QA_flag == 0: return False  # no glints
        
        ifRead = True
        if self.today is None: 
            self.today = today
        else:
            if self.today.strftime(self.FNm_templ) == today.strftime(self.FNm_templ):
                ifRead = False
            else:
                self.today = today
        if ifRead:
            self.from_nc_data(self.today.strftime(self.FNm_templ), self.log)
        #
        fx, fy = self.grid.geo_to_grid(lons, lats)
        ix = np.round(fx).astype(np.int32)
        iy = np.round(fy).astype(np.int32)
        ii = ix + iy * self.grid.nx   # indices of fires in 1D
        idx_sort = np.argsort(ii)    # indices that sort the given fires
        idxGlint = np.searchsorted(self.i1d, ii[idx_sort])  # will it work?????????
        ifGlint = ii[idxGlint] == ii[idx_sort]
        return ifGlint

    #=====================================================================

    def find_glints(self, lons, lats, szas, FRPs, datesIn, satellites, grid, revisit_days, chFNmOut, 
                    mpirank, mpisize, comm, ifCleanSpace):
        #
        # Check the time series for same-grid-cell fires every revisit period
        # tsMatrix cannot be used because times are not strictly monotonic
        # Neither gradual accumulation is easy: we do not know how many glints will be found.
        # So, just go cell-by-cell writing glints to the file.
        #
        self.grid = grid[0]
        self.gridNm = grid[1]

        ifGlint = szas < 90  # glint can happen only during day
        noGlint = szas >= 90  # A nighttime registration means that the cell is not glint
        #
        # project all daytime fires to the given grid
        #
        fx, fy = self.grid.geo_to_grid(lons, lats)
        ix = np.round(fx).astype(np.int32)
        iy = np.round(fy).astype(np.int32)
        i1D = ix + iy * self.grid.nx
        idxSortIn = np.argsort(i1D)  # index that would sort all input vectors along lon-lat 1D index
        datesSort = datesIn[idxSortIn]
        #
        # VIIRS has a feature that two satellites appear at the same position one after another
        # with interval of exactly half of the revisit time. MODIS does not have this feature.  
        #
        satSort = satellites[idxSortIn]
#        satTypes = 
        ifFlip = len(list(set(satellites))) == 2   # if two satellites, they should flip every half-revisit time
        if ifFlip:
            revDays_loc = revisit_days // 2
        else:
            revDays_loc = revisit_days
        #
        # Get the list of fire-prone pixels without repetitions, removing those grid cells where
        # nighttime fire registrations were present.
        #
        ixyUnique = np.array(sorted(list(set(i1D[ifGlint]) - set(i1D[noGlint]))), dtype=np.int64)
        #
        # Get the 1-D mask: 1D indices of locations with glints possible. Dates can be repetitive
        #
        idxGlint = np.searchsorted(i1D[idxSortIn], ixyUnique)
        #
        # list of times as dates in a regular scale
        #
        datesAll = datesSort[idxGlint]   #dates are unsorted & repetitive: sorting is geogragphical, not temporal
        tStart = np.min(datesAll)
        tEnd = np.max(datesAll)
        nDays = int((tEnd - tStart)/spp.one_day)
        dayLst = np.array(list((tStart + spp.one_day * i for i in range(nDays+1))))     # regular list
        self.years = np.array(range(tStart.year, tEnd.year+1))
        #
        # each grid cell is processed one by one
        #
        cnt=0
        nGlints = 0
        _i1d = []
        _lons = []
        _lats = []
        _nFires = []
        _frps = []
        _chDates = []
        _satellites = []
        
        for ii in ixyUnique:
            cnt += 1
            if np.mod(cnt-1, mpisize) != mpirank: continue
            if np.mod(cnt-1, 1001) == 0: print('MPI %g check cell %i, i1D %i' % (mpirank, cnt, ii))
            #
            # Get all cases with fires in this cell
            #
            idxFiresInCell = i1D[idxSortIn] == ii
            if np.sum(idxFiresInCell) < 2: continue

            fireDates = sorted(list(set(datesSort[idxFiresInCell])))
            if self.method == 'mask':
                if len(fireDates) < 3 or len(fireDates) > nDays / revDays_loc + 2:
                    continue  # too few or too many fires to be a regular glint
            else: raise ValueError('Unknown glint method')
            
            idxFireDates = np.searchsorted(dayLst, fireDates)
            #
            # Get the 0-1 for fire-no-fire for all days
            #
            iFire = np.zeros(shape=(len(dayLst)), dtype=np.int8)
            iFire[idxFireDates] = 1    # fires durig these days
            #
            # check that fires are present only at every revisit-th day
            #
            idxFirstFire = np.argmax(iFire)  # returns the first occurence of max of the array, index of the first 1
            mask = np.zeros(shape=(len(dayLst)), dtype=np.int8)
            mask[idxFirstFire::revDays_loc] = 1  # this mask == 1 every revisit-th day starting from the first fire
            # glint?
            if np.sum(mask * iFire) > 2 and np.sum((1-mask) * iFire) == 0:
                FRPtot = np.sum(FRPs[idxSortIn][idxFiresInCell])
                iTmp = np.mod(ii, self.grid.nx)
                jTmp = (ii-iTmp)/self.grid.nx
                llon, llat = self.grid.grid_to_geo(iTmp, jTmp)
                _sats = satellites[idxSortIn][idxFiresInCell]
                # satellite IDs must be the same over the revisit time
                # or flipping if there are two, then revTime_loc = revisit_days / 2
                if ifFlip:  # two satellites, flipping revisit
                    for igl, glDate in enumerate(fireDates):
                        if _sats[igl] == _sats[0]: 
                            if np.mod((glDate - fireDates[0]).total_seconds() / 86400, revisit_days) != 0: 
                                ifGlint = False
                                break    # same satellite but not full-revisit time
                        else:
                            if np.mod((glDate - fireDates[0]).total_seconds() / 86400, revisit_days) == 0: 
                                ifGlint = False
                                break  # different satellite but full-revisit time
                    if not ifGlint: continue
                else:
                    if not np.all(_sats == _sats[0]):
                        continue    # not a glint: different satellites see the glint

                
                    
                _i1d.append(ii)
                _lons.append(llon)
                _lats.append(llat)
                _nFires.append(np.sum(iFire))
                _frps.append(FRPtot)
                _chDates.append(fireDates)
                self.log.log('glint = %i %i %i %g %g %i %g  %s' %
                             (ii, iTmp, jTmp, llon, llat, np.sum(iFire), FRPtot, str(fireDates)))
                nGlints += 1

            if nGlints > 2: break

#        self.log.log('END_LIST = glints')
        self.i1d = np.array(_i1d)
        self.lons = np.array(_lons)
        self.lats = np.array(_lats)
        self.nFires = np.array(_nFires)
        self.frps = np.array(_frps)
        self.chDates = np.array(_chDates, dtype=object)
        #
        # Store the findings, to temporary MPI file
        #
        self.glints_to_txt(chFNmOut + '_MPI%03i.txt' % mpirank) # write the intermediate MPI file
        #
        # Require all MPIs to make their output
        #
        spp.MPI_join('glints1_%s' % self.years, os.path.join(os.path.split(chFNmOut)[0],'tmp'),
                     mpisize, mpirank, comm)
        #
        # zero-MPI member will collect all MPI outputs 
        #
        if mpirank == 0:
            #
            # Scan the other MPI files collecting their info. We will swallow the complete 
            # list of glints - it cannot be overly large
            #
            for iMPI in range(mpisize):
                nlGrp = namelist.NamelistGroup.fromfile(chFNmOut + '_MPI%03i.txt' % iMPI)
                if iMPI == 0: nlGlints = nlGrp.get('glint')
                else: nlGlints.extend(nlGrp.get('glint'))
                #
                # Delete the temporary file
                #
                if ifCleanSpace: os.remove(chFNmOut + '_MPI%03i.txt' % iMPI)
            #
            # sort the glint locations following i1d index
            #
            glints = np.array(nlGlints.values()[0])
            i1d = np.array(list((np.int64(g.split()[0]) for g in glints)), dtype=np.int64)
            idxSort = np.argsort(i1d)   # sort a 1D index of the (ix, iy) pairs
            #
            # Set all elements overwriting the ones we have in self
            #
            self.i1d = i1d[idxSort]
            self.lons = np.array(list((np.float32(g.split()[3]) for g in glints[idxSort])), dtype=np.float32)
            self.lats = np.array(list((np.float32(g.split()[4]) for g in glints[idxSort])), dtype=np.float32)
            self.nFires = np.array(list((int(g.split()[5]) for g in glints[idxSort])), dtype=int)
            self.frps = np.array(list((np.float32(g.split()[6]) for g in glints[idxSort])), dtype=np.float32)
            if self.method == 'mask':
#                self.chDates = np.array(list((g[g.index('[') : g.index(']')+1]) for g in glints[idxSort]))
#                self.chDates = np.array(list((dt.datetime.strptime(gg.strip(),'%Y%m%d_%H%M') 
#                                              for gg in list((g.split('[')[1].split(']')[0].split(',') 
#                                                             for g in glints)))))
                self.chDates = np.array(list((list((dt.datetime.strptime(gg.strip(),'%Y%m%d_%H%M')
                                                    for gg in g.split('[')[1].split(']')[0].split(',')))
                                              for g in glints)))
            else: raise ValueError('Unknown method 2:' + self.method)
            #
            # Write the final output in all formats we have
            #
            self.glints_to_txt(chFNmOut)
            self.glints_to_kml(chFNmOut + '.kml')
            self.glints_to_nc_data(chFNmOut + '.nc4')
        #
        # just to make things clean
        #
        spp.MPI_join('glints2_%s' % self.years, os.path.join(os.path.split(chFNmOut)[0],'tmp'),
                     mpisize, mpirank, comm)


    #====================================================================

    def subset(self, ifGoodGlint):
        #
        # Takes a subset of existing glints using the boolean mask of good ones
        #
        glNew = QA_glints(self.chGlintFNmTempl, self.method, self.log)
        glNew.grid = self.grid
        glNew.gridNm = self.gridNm
        glNew.i1d = self.i1d[ifGoodGlint]
        glNew.lons = self.lons[ifGoodGlint]
        glNew.lats = self.lats[ifGoodGlint]
        glNew.nFires = self.nFires[ifGoodGlint]
        glNew.frps = self.frps[ifGoodGlint]
        if self.method == 'correlation':
            glNew.cRev_minus_1 = self.cRev_minus_1[ifGoodGlint]
            glNew.cRevisit = self.cRevisit[ifGoodGlint]
            glNew.cRev_plus_1 = self.cRev_plus_1[ifGoodGlint]
        elif glNew.method == 'mask':
            glNew.chDates = self.chDates[ifGoodGlint]
        return glNew


    #====================================================================
    
    def glints_to_txt(self, chFNmOut):
        #
        # Writes the glint object to the given file
        #
        if self.method == 'mask':
            self.to_txt(chFNmOut, self.glint_string, 'glint', self.years,
                        np.array(list(('[' + ', '.join(list((str(s2.strftime('%Y%m%d_%H%M')) 
                                                             for s2 in s))) + ']' 
                                       for s in self.chDates))))
        else: raise ValueError('Unsupported method: ' + self.method)


    #====================================================================
    
    def glints_to_kml(self, chFNmOut):
        #
        # Stores the list of glints into a list of "My places" for GoogleEarth
        #
        self.to_kml(chFNmOut, '%s, glints, year(s) %s' % (self.gridNm, self.years),
                    'GL', self.years, 'dates',                          # abbreviation, years, name of the extra description
                    list((', '.join(list((s.strftime('%Y%m%d_%H%M') for s in gl ))) for gl in self.chDates)))  # extra description, point-specific


    #=======================================================================================

    def glints_to_nc(self, chFNmOut):
        #
        # Store the glints to netcdf. Note that netcdf requires ix and iy, not i1d
        #
        self.ix = np.mod(self.i1d, self.grid.nx).astype(np.int32)
        self.iy = ((self.i1d - self.ix) / self.grid.nx).astype(np.int32)
        self.to_nc(chFNmOut,
                   {'file_type' : 'glints', 'criteria' : self.method,
                    'grid_name' : self.gridNm,
                    'year' : self.years})    #nlGrpGlints.get('parameters').get_uniq('years')})

    #===========================================================================

    def report(self):
        #
        try:
            return 'Glints %s %s %s' % (self.method, self.gridNm, self.years)
        except:
            return 'Glints: no QA'

#################################################################################

class QA():
    #
    # Overarching class for all above
    #
    def __init__(self, QA_action, QA_overlap_scans, QA_frequent_fires, QA_glint, QA_sml_huge_fires, QA_void_fires, QA_problematic_days, chDirMetadata, log):
        #
        # Initialize all above classes
        #
        self.QA_action = QA_action
        self.QA_overlap_scans = QA_overlap_scans
        self.QA_frequent_fires = QA_frequent_fires
        self.QA_glint = QA_glint
        self.QA_sml_huge_fires = QA_sml_huge_fires 
        self.QA_void_fires = QA_void_fires 
        self.QA_problematic_days = QA_problematic_days
        self.chDirMetadata = chDirMetadata
        self.log = log

    #=============================================================================
    @classmethod
    def from_params(cls, QA_action, ifMergeOverlappingScans, chFrequentFiresTempl, chGlintTempl, chVoidLU_label, all_LUs, minFRP_MW, maxFRP_MW, sources, chDirMetadata, log):
        #
        # Initialize all above classes
        #
        # What is expected from QA
        self.QA_action = QA_action
        # scan overlaps
        if ifMergeOverlappingScans and self.QA_action == 'FULL':
            QA_so = QA_overlap_scans(np.int64(1))
        else: 
            QA_so = None
        # frequent fires
        if chFrequentFiresTempl is None or self.QA_action != 'FULL': 
            QA_ff = None
        else:
            QA_ff = QA_frequent_fires(chFrequentFiresTempl, log)
        # glints
        if chGlintTempl is None or self.QA_action != 'FULL':
            QA_gl = None
        else:
            QA_gl = QA_glints('mask', log)
        # hige FRP
        if maxFRP_MW is None  or self.QA_action != 'FULL':
            QA_hf = None
        else:
            QA_hf = QA_sml_huge_fires(minFRP_MW, maxFRP_MW)
        # void LU
        if chVoidLU_label is None  or self.QA_action != 'FULL':
            QA_vf = None
        else:
            QA_vf = QA_void_fires(chVoidLU_label, all_LUs)
        # bad days
        if self.QA_action == 'BAD_DAYS' or self.QA_action == 'FULL':
            QA_pd = QA_problematic_days(sources)  # sources can be none

        return cls(QA_so, QA_ff, QA_gl, QA_hf, QA_vf, QA_pd, chDirMetadata, log)


    #===============================================================
    @classmethod
    def from_control_namelist(cls, nlIni, all_LUs, chDirMetadata):
        #
        # In the IS4FIRES v.3.0 control file huge fires are represented
        # via the max FRP allowed
        #
        QAlog = spp.log(nlIni.get_uniq('QA_log'))
        QA_action = nlIni.get_uniq('QA_action')
        if QA_action == 'NONE':
            print('QA disabled')
            return cls.void()
        # scan overlaps
        if QA_action == 'FULL':
            QA_so = QA_overlap_scans(QAlog)
            # frequent fires
            QA_ff = QA_frequent_fires.from_control_namelist(nlIni, chDirMetadata) #, QAlog)
            # glints
            QA_gl = QA_glints.from_control_namelist(nlIni, chDirMetadata, QAlog)
            # hige FRP
            QA_hf = QA_sml_huge_fires.from_control_namelist(nlIni)
            # void LU
            QA_vf = QA_void_fires.from_control_namelist(nlIni, all_LUs)
        else:
            QA_so = None
            QA_ff = None
            QA_gl = None
            QA_hf = None
            QA_vf = None
        # bad days
        if QA_action == 'FULL' or QA_action == 'BAD_DAYS':
            QA_pd = QA_problematic_days.from_control_namelist(nlIni)
        
        return cls(QA_action, QA_so, QA_ff, QA_gl, QA_hf, QA_vf, QA_pd, chDirMetadata, QAlog)

    #=================================================================================
    @classmethod
    def from_nc(cls, fIn_, log):
        #
        # Need to open?
        if isinstance(fIn_, str): fIn = nc4.Dataset(fIn_,'r')
        else: fIn = fIn_
        #
        QA_action = fIn.QA_action
        #
        # Retrieve those classes one by one
        # Overlapping scans
        QA_so = QA_overlap_scans.from_nc(fIn, log)
        # frequent fires
        QA_ff = QA_frequent_fires.from_nc_QA(fIn, log)
        # glints
        QA_gl = QA_glints.from_nc_QA(fIn, log)
        # hige FRP
        QA_hf = QA_sml_huge_fires.from_nc(fIn, log)
        # void LU
        QA_vf = QA_void_fires.from_nc(fIn, log)
        # bad days
        QA_pd = QA_problematic_days.from_nc(fIn, log)
        
        return cls(QA_action, QA_so, QA_ff, QA_gl, QA_hf, QA_vf, QA_pd, None, log)

    #=================================================================================
    
    def to_nc(self, fOut_):
        #
        # Need to open?
        if isinstance(fOut_, str): fOut = nc4.Dataset(fOut_,'w')
        else: fOut = fOut_
        #
        fOut.QA_action = self.QA_action
        fOut.QA_flag = self.get_flag()
        #
        # Wrote classes one by one
        # Overlapping scans
        if self.QA_overlap_scans is not None: self.QA_overlap_scans.to_nc(fOut)
        # frequent fires
        if self.QA_frequent_fires is not None: self.QA_frequent_fires.to_nc_QA(fOut)
        # glints
        if self.QA_glint is not None: self.QA_glint.to_nc_QA(fOut)
        # hige FRP
        if self.QA_sml_huge_fires is not None: self.QA_sml_huge_fires.to_nc(fOut)
        # void LU
        if self.QA_void_fires is not None: self.QA_void_fires.to_nc(fOut)
        # bad days
        if self.QA_problematic_days is not None: self.QA_problematic_days.to_nc(fOut)
        
    #=================================================================================
    @classmethod
    def void(cls):
        #
        # Nullify all above classes
        #
        return cls(None, None, None, None, None, None, None)

    #=================================================================================

    def report(self):
        #
        # Reports in one line what is done as QA
        #
        strRep = 'QA report:' + self.QA_action + ',\n'
        #
        # overlapping pixels
        strRep += self.QA_overlap_scans.report() + ',\n'
        # frequent fires
        strRep += self.QA_frequent_fires.report() + ',\n'
        # glints
        strRep += self.QA_glint.report() + ',\n'
        # hige FRP
        strRep += self.QA_sml_huge_fires.report() + ',\n'
        # void LU
        strRep += self.QA_void_fires.report() + ',\n'
        # bad days
        strRep += self.QA_problematic_days.report()
        
        return strRep 

    #======================================================================

    def get_mask(self, lons, lats, FRPs, LUs, day, QA_done):  # QA_done is the flag of QA that has been already applied
        #
        # Bad-fires mask is an envelope of all
        # Overlapping scans have been taken care of - they do not remove but scale the FRP values
        #
        mask = np.zeros(shape=FRPs.shape, dtype=np.byte)
        #
        # BAD days are not about mask, they disqualify the whole day. Use if_bad_day for that. Here we do not check it
        # 
#        if self.QA_problematic_days is not None:
#            if self.QA_problematic_days.QA_flag > 0 and QA_done.QA_problematic_days.QA_flag == 0:
#                mask = self.QA_problematic_days.get_mask(day)
#                if mask: return mask   # bad day, nothing to check further
        # glints
        if self.QA_glint is not None:
            if self.QA_glint.QA_flag > 0 and QA_done.QA_glint.QA_flag == 0:
                mask += self.QA_glint.get_mask(lons,lats, day)
        # small and huge fires
        if self.QA_sml_huge_fires is not None:
            if self.QA_sml_huge_fires.QA_flag > 0 and QA_done.QA_sml_huge_fires.QA_flag == 0:
                mask += self.QA_sml_huge_fires.get_mask(FRPs)
        # void LU
        if self.QA_void_fires is not None:
            if self.QA_void_fires.QA_flag > 0 and QA_done.QA_void_fires.QA_flag == 0:
                mask += self.QA_void_fires.get_mask(LUs)
        # frequent fires
        if self.QA_frequent_fires is not None:
            if self.QA_frequent_fires.QA_flag > 0 and QA_done.QA_frequent_fires.QA_flag == 0:
                mask += self.QA_frequent_fires.get_mask(lons, lats, day)
#        if np.sum(mask) > 0: self.log.log(day.strftime('%Y%m%d:') + ' masked %i out of %i fires' % 
#                                          (np.sum(mask), len(mask)))
        return mask

    #======================================================================

    def if_bad_day(self, today):
        return self.QA_problematic_days.if_bad_day(today)
    
    #======================================================================
    
    def get_flag(self):
        QA_flag = def_maxval_flag    # set the zeroes and the leading number 5
        if self.QA_overlap_scans is not None: QA_flag += self.QA_overlap_scans.QA_flag
        if self.QA_problematic_days is not None: QA_flag += self.QA_problematic_days.QA_flag 
        if self.QA_glint is not None: QA_flag += self.QA_glint.QA_flag
        if self.QA_sml_huge_fires is not None: QA_flag += self.QA_sml_huge_fires.QA_flag
        if self.QA_void_fires is not None: QA_flag += self.QA_void_fires.QA_flag
        if self.QA_frequent_fires is not None: QA_flag += self.QA_frequent_fires.QA_flag 
        return QA_flag
        
    #======================================================================
    
    def report_flag(self, log=None):
        flag = self.get_flag()
        sTmp = 'QA flag = %i\n' % flag
        for f, chF in [(def_scan_overlap_flag, 'scan overlap'),
                       (def_problematic_days_flag, 'problematic days'),
                       (def_sml_huge_fires_flag,'small-huge fires'),
                       (def_void_fires_flag,'void fires'),
                       (def_frequent_fires_flag,'frequent fires'),
                       (def_glints_flag,'glints')]:
            if flag // f - (flag // (f * 10) * 10) == 0: sTmp += '%s is OFF, \n' % chF
            else: sTmp += '%s is ON, \n' % chF
        if log is not None: log.log(sTmp + '\n')
        return sTmp

    #=======================================================================
    
    def add(self, QA_add):
        #
        # Returns the QA object that includes own featrues and, where they were missing, new ones.
        #
        # overlapping scans
        if self.QA_overlap_scans is None: self.QA_overlap_scans = copy.copy(QA_add.QA_overlap_scans)
        # glints
        if self.QA_glint is None: self.QA_glint = copy.copy(QA_add.QA_glint)
        # small and huge fires
        if self.QA_sml_huge_fires is None: self.QA_sml_huge_fires = copy.copy(QA_add.QA_sml_huge_fires)
        # void LU
        if self.QA_void_fires is None: self.QA_void_fires = copy.copy(QA_add.QA_void_fires) 
        # frequent fires
        if self.QA_frequent_fires is None: self.QA_frequent_fires = copy.copy(QA_add.QA_frequent_fires )
        # bad days
        if self.QA_problematic_days is None: self.QA_problematic_days = copy.copy(QA_add.QA_problematic_days)



#################################################################################
#################################################################################
#
# Verify various stages of processing
#
#################################################################################
#################################################################################

def verify_tsM_map_predicted(tsmLst, land_use, grid, mapDir, mapFNmTempl):
    #
    # Generates the output file from the coarse input matrix and compares
    # time series summing-up the detailed files. 
    #
    for iTSM, tsm in enumerate(tsmLst):
        #
        # Check specific cell
        #
        if land_use.LUtypes[iTSM] == 'NA_temperate_forest':
            ts2check = tsm
            break 
    for ist, st in enumerate(tsm.stations):
        print(ist, st.code, st.lon-360, st.lat)
    print()
    for itime, time in enumerate(tsm.times):
        print(time, tsm.vals[itime,46])
    

#======================================================================

def verify_tsMatrices(chFNmTemplate, log):
    #
    # Prints statistics for the tsMatrices written in the corresponding directory
    #
    for chFIn in glob.glob(chFNmTemplate):
        MyTimeVars.TsMatrix.verify(chFIn, log)


#=======================================================================================

def multi_resolution_check(chFNmHiResTempl, chFNmLowResTempl, chFNmOutTempl, years, log):
    #
    # Takes two lists of glints based on grids with different resolutions, then
    # uses the coarser-grid list to shut down fine-grid glints.
    # The idea is that a fine-grid glint detection can be accidental if the region 
    # is heavily burning. But if so, a coraser-grid list will not have it because
    # coarser grid cell will include more fires making the accidental satisfaction
    # of the selection criterion (possible occurance only every revisit time) less probable
    #
    for year in years:
        glHiRes = QA_glints('mask', log)
        glHiRes.read_glints_txt(chFNmHiResTempl % year)
        glLowRes = QA_glints('mask', log)
        glLowRes.read_glints_txt(chFNmLowResTempl % year)
        #
        # Need to project the high-res glints to the low-res grid, then retain 
        # only those high-res glints that are visible in the low-res list.
        #
        fx, fy = glLowRes.grid.geo_to_grid(glHiRes.lons, glHiRes.lats)
        ix = np.round(fx).astype(int)
        iy = np.round(fy).astype(int)
        ii = ix + iy * glLowRes.grid.nx
        # index of low-resolution glints in the high-resolution glint list
        idxHL = np.searchsorted(glLowRes.i1d, ii)
        ifConfirmedGlint = ii == glLowRes.i1d[idxHL]
        # Remove the non-confirmed glints from the high-resolution list
        glNew = glHiRes.subset(np.logical_not(ifConfirmedGlint))
        # Store
        chDirOut, chFNmOut = os.path.split(chFNmOutTempl % year)
        
        glNew.glints_to_txt(chDirOut, chFNmOut)
        glNew.glints_to_kml(chDirOut, chFNmOut, chFNmOut + '.kml')
        glNew.glints_to_nc(namelist.NamelistGroup.fromfile(chFNmOutTempl % year),
                           chDirOut, chFNmOut + '.kml')

        
        


#############################################################################
#############################################################################

if __name__ == '__main__':

    ifFreqFires_from_raw = False
    ifFreqFires_from_fire_rec = False
    convert_5w_2_FF_class = False
    draw_FF_maps = False
    ifMultiCriteriaFFMaps = False
    ifApplyMask = True
    ifCheckTSM = False
    
    #
    # New frequent fires identification procedure
    #
    # Windows
    chProcessedMODIS_main = 'd:\\results\\fires\\IS4FIRES_v3_0_grid_FP_2024'
    dirOut = 'd:\\results\\fires\\frequent_fires'
    
    # Puhti
#    chProcessedMODIS_main = '/fmi/scratch/project_2001411/fires/IS4FIRES_v3_0_grid_FP'
#    dirOut = '/fmi/scratch/project_2001411/fires/frequent_fires'

#    nFires_main = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]  #
    nFires_main = [20] #, 100, 110]  #
#    nYears_main = [1, 2, 3, 4, 5, 6, 7] 
    nYears_main = [4] 

    criteria = [(20,5),(90,1)]  # 20 fires per year x 5 years OR 90 fires x 1 year
#    criteria = [(20,5),(100,1)]  # 20 fires per year x 5 years OR 100 fires x 1 year
#    criteria = [(20,5),(110,1)]  # 20 fires per year x 5 years OR 110 fires x 1 year
    chCriteria = '_'.join((('nF%i_nY%i' % (setCrit[0], setCrit[1]) for setCrit in criteria)))
#    chCriteria = ''
#    for setCrit in criteria:
#        chCriteria += '_nF%i_nY%i' % (setCrit[0], setCrit[1])

    #----------------------------------------------------------------------
    if ifFreqFires_from_raw:
        # From the raw fires
        # Initialize the object but do not give the FF-list file name
        FFnew = QA_frequent_fires(None, None, spp.log())
        # Now, find them
        FFnew.find_frequent_fires_from_daily_maps(os.path.join(chProcessedMODIS_main, 'glob_3_0_LST', 
                                                               'IS4FIRES_v3_0_glob_3_0_%Y%m%d_LST.nc4'), 
                                                  'w:\\fires\\grd_2024_glob_0_03.txt',
                                                  dt.datetime(2000,3,1),dt.datetime(2023,12,31), 
                                                  nFires_main, nYears_main,          # FF_thresh_cases_per_yr, FF_thresh_yrs
                                                  dirOut)

    #----------------------------------------------------------------------
    if ifFreqFires_from_fire_rec:
        # From intermediate fire map
        # Initialize the object but do not give the FF-list file name
        FFnew = QA_frequent_fires(None, None, spp.log())
        # Now, find them
        FFnew.find_frequent_fires_from_fire_records(
                                        os.path.join(dirOut,'fire_map_2000-2023.nc4'),
                                        dt.datetime(2000,3,1), dt.datetime(2023,12,31), 
                                        nFires_main, nYears_main,   # FF_thresh_cases_per_yr, FF_thresh_yrs
                                        dirOut)

    #----------------------------------------------------------------------
    if convert_5w_2_FF_class:
        # Initialize the object but do not give the FF-list file name
        # Get the 5w frequent fires
        FFnew = QA_frequent_fires(None, None, spp.log())
        for yr in range(2002,2020):
            FFnew.convert_5wFF_to_FF(yr,
                                     'd:\\project\\Fires\\make_yearly_mask\\masks-5w\\Mask_21-%i.nc' % yr, # input
                                     'd:\\results\\Fires\\FF_from_5w')            # output dir
            # re-read for drawing
            FFnew2 = QA_frequent_fires(None,
                                       'd:\\results\\Fires\\FF_from_5w\\FreqFires_from_5w_Mask_21-%i.nc4' % yr,
                                       spp.log())
            FFnew2.draw_FF_maps('d:\\results\\Fires\\FF_from_5w\\pics_21')
        for yr in range(2017,2023):
            FFnew.convert_5wFF_to_FF(yr,
                                     'd:\\project\\Fires\\make_yearly_mask\\masks-5w\\Mask_20-oper-%i.nc' % yr, # input
                                     'd:\\results\\Fires\\FF_from_5w')            # output dir
            # re-read for drawing
            FFnew2 = QA_frequent_fires(None,
                                       'd:\\results\\Fires\\FF_from_5w\\FreqFires_from_5w_Mask_20-oper-%i.nc4' % yr,
                                       spp.log())
            FFnew2.draw_FF_maps('d:\\results\\Fires\\FF_from_5w\\pics_20-oper')

    #----------------------------------------------------------------------
    if draw_FF_maps:
        for iNFires, nFires in enumerate(nFires_main):
            for iNYears, nYears in enumerate(nYears_main):
                for iYr, yr in enumerate(range(2000,2024)):
                    # Initialise the FF-object by reading the pre-recorded nc file
                    try:
                        FFnew = QA_frequent_fires(None,
                                                  os.path.join(dirOut,
                                                               'Frequent_fires_ann_%i_nYrs_%i_%i.nc4'
                                                               % (nFires_main[iNFires],
                                                                  nYears_main[iNYears], yr)),
                                                  spp.log())
                    except:
                        print('######### No file found: ', os.path.join(dirOut,
                                                               'Frequent_fires_ann_%i_nYrs_%i_%i.nc4'
                                                               % (nFires_main[iNFires],
                                                                  nYears_main[iNYears], yr)))
                        continue
                    # draw FRP and nFires
                    spp.ensure_directory_MPI(os.path.join(dirOut,'pics'))
                    FFnew.draw_annual_FF_map(nFires_main[iNFires], nYears_main[iNYears], yr, 
                                             os.path.join(dirOut,'pics'))

    #----------------------------------------------------------------------
    if ifMultiCriteriaFFMaps:
        for iYr, yr in enumerate(range(2000,2024)):
            FFmain = QA_frequent_fires(None, None, spp.log())
            chCrit = ''
            for setCrit in criteria:
                # Initialise the FF-object by reading the pre-recorded nc file
                FFnew = QA_frequent_fires(None,
                                          os.path.join(dirOut,
                                                       'Frequent_fires_ann_%i_nYrs_%i_%i.nc4'
                                                       % (setCrit[0], setCrit[1], yr)),
                                          spp.log())
                FFmain.add(FFnew)
                chCrit += '_nF%i_nY%i' % (setCrit[0], setCrit[1])
            
            FFmain.frequent_fires_to_text_file(dirOut, [yr],
                                               'Frequent_fires_multicrit_%s_%i.txt') # % (chCrit, yr))
            FFmain.frequent_fires_to_nc(dirOut, [yr],
                                        'Frequent_fires_multicrit_%s_%i.nc4') # % (chCrit, yr))
            FFmain.frequent_fires_to_kml(dirOut, [yr],
                                         'Frequent_fires_multicrit_%s_%i.kml') # % (chCrit, yr))
            FFmain.draw_annual_FF_map(chCrit, None, yr, os.path.join(dirOut,'pics'))
        
    #----------------------------------------------------------------------

    if ifApplyMask:
#        dirIn = 'v:\\andreas\\AirQast\\emis\\fires\\masked_fires_5w'
#        dirOut = 'v:\\andreas\\AirQast\\emis\\fires\\masked_fires_FF_n20_y5_n90_y1'
#        dirIn = 'c:\\MODIS_raw_data\\masked_fires_5w'
#        dirOut = 'c:\\MODIS_raw_data\\masked_fires_FF_%s' % chCriteria
        
        dirIn = 'u:\\silam2\\input\\emis\\IS4FIRES_v2_0\\masked_fires_5w'
        dirOut = 'u:\\silam2\\input\\emis\\IS4FIRES_v2_0\\masked_fires_%s' % chCriteria
        mask_files = 'd:\\results\\fires\\frequent_fires\\Frequent_fires_multicrit_' + chCriteria + '_%i.nc4'
        inFiles = glob.glob(os.path.join(dirIn,'*_2019*frpd'))
        print('Processing %i files' % len(inFiles))
        spp.ensure_directory_MPI(dirOut)
        log_main = spp.log(os.path.join(dirOut,'__summary.log'))
        log_main.log('Summary of frequent fire removal, unit=[MW]\n')
        # just basic
        FF_mask = None
        year2process = 0
        FRP_old_masked = {}
        FRP_new_masked = {}
        FRP_total = {}
        
        for inFNm in inFiles:
            with open(os.path.join(dirIn, os.path.split(inFNm)[-1]), 'rt') as inf:
                with open(os.path.join(dirOut, os.path.split(inFNm)[-1]), 'wt') as outf:
                    outf.write("# Remasking applied. Input file:\n # %s\n" % inFNm)
                    outf.write("# Mask files: %s\n" % mask_files)
                    outf.write("\n" )
                    for l in inf:
                        ifOld = False
                        ifNew = False 
                        if l.startswith("##$$##$$##  fire ="): ## Unmask
                            l = l[12:]
                            ifOld = True
                        if l.startswith("### fire ="):
                            l = l[4:]
                            ifOld = True
                        if l.startswith("fire ="):
                            a = l.split()
                            flon = float(a[9])
                            flat = float(a[10])
                            year = int(a[3])
                            FRP = float(a[14])
                            if year != year2process:
                                FF_mask = QA_frequent_fires(None, mask_files % year, spp.log())
                                if year2process > 0:
                                    log_main.log('Year = %i, FRP_old_mask = %i, FRP_new_mask = %i, FRP_total = %i\n' %
                                                 (year2process, FRP_old_masked[year2process], 
                                                  FRP_new_masked[year2process], FRP_total[year2process]))
                                year2process = year
                                FRP_old_masked[year] = 0.
                                FRP_new_masked[year] = 0.
                                FRP_total[year] = 0.
                            if FF_mask.mask_given_locations(np.array([flon]), np.array([flat]), year):
                                outf.write("###$$### ")
                                ifNew = True
                            if ifOld:
                                FRP_old_masked[year] += FRP
                            elif ifNew:
                                FRP_new_masked[year] += FRP
                            else:  
                                FRP_total[year] += FRP
                        outf.write(l)

    #----------------------------------------------------------------------
    #
    # Generic check of tsMatrices
    #
#    chInDir = 'f:\\project\\fires\\forecasting_v3_0\\output_cell_mdls\\glob_3_0_meteo_CESM_HIST'
    chInDir = '/fmi/scratch/project_2001411/fires/forecasting_v3_0/output_cell_mdls/glob_3_0_meteo_CESM_HIST'
    if ifCheckTSM:
        verify_tsMatrices(os.path.join(chInDir,'*.nc4'), 
                          spp.log(os.path.join(chInDir,'..','tsM_check.log')))

