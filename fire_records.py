'''

 Class fire_records handles a vector of fire records with non-zero FRP
 It can read, write, and manipulate them 

Created on 22.12.2025

@author: sofievm
'''

import numpy as np, datetime as dt
import copy
import netCDF4 as nc4
from toolbox import silamfile, supplementary as spp


################################################################################################

class fire_records():

    #========================================================================

    def __init__(self, log):
        self.nFires = 0      # empty object
        self.timezone = None              # can be UTC or solar local time
        self.timeStart = None
        self.grid = None
        self.sorted = None
        self.LU_metadata = None
#        self.QA_flag = None  # not None but undefined
        self.log=log

    #========================================================================

    def init_data_structures(self, nRec, grid=None, gridName=None):
        #
        # Reserves space for the expected number of fires (can be zero)
        #
        self.FRP = np.ones(shape=(nRec), dtype=np.float32)
        self.lon = np.ones(shape=(nRec), dtype=np.float32)
        self.lat = np.ones(shape=(nRec), dtype=np.float32)
        self.dS = np.ones(shape=(nRec), dtype=np.float32)
        self.dT = np.ones(shape=(nRec), dtype=np.float32)
        self.T4 = np.ones(shape=(nRec), dtype=np.float32)
        self.T4b = np.ones(shape=(nRec), dtype=np.float32)
        self.T11 = np.ones(shape=(nRec), dtype=np.float32)
        self.T11b = np.ones(shape=(nRec), dtype=np.float32)
        self.TA = np.ones(shape=(nRec), dtype=np.float32)
        self.ix = np.ones(shape=(nRec), dtype=np.int32)  # indices in the grid of the maps
        self.iy = np.ones(shape=(nRec), dtype=np.int32)  # indices in the grid of the maps
        self.time = np.ones(shape=(nRec), dtype=np.int64)  #object)
        self.line = np.ones(shape=(nRec), dtype=np.int16)
        self.sample = np.ones(shape=(nRec), dtype=np.int16)
        self.SolZenAng = np.ones(shape=(nRec), dtype=np.int16)
        self.ViewZenAng = np.ones(shape=(nRec), dtype=np.int16)
        self.LU = np.ones(shape=(nRec), dtype=np.int16)
        self.satellite = np.ones(shape=(nRec), dtype='|S1')
        self.grid = copy.deepcopy(grid)   # main grid of the analysis
        self.gridName = gridName          # name of the grid (no spaces)
        self.timezone = None    # can be UTC, solar local time,...
        self.nFires = 0  # yes, initialized, but no fires recorded, yet

    #========================================================================

    def from_dic(self, FP_dic):
        #
        # creates the object from dictionary that should have all vectors
        #
        self.lon = FP_dic['lon'] 
        self.lat = FP_dic['lat']
        self.time = FP_dic['time']  # assumed daytime object
        self.FRP = FP_dic['FRP']
        self.dS = FP_dic['dS'] 
        self.dT = FP_dic['dT'] 
        self.SolZenAng = FP_dic['sza']
        self.ViewZenAng = FP_dic['ViewZenAng']
        self.T4 = FP_dic['T4'] 
        self.T4b = FP_dic['T4b'] 
        self.T11 = FP_dic['T11'] 
        self.T11b = FP_dic['T11b']
        self.TA = FP_dic['TA'] 
        self.ix = FP_dic['ix'] 
        self.iy = FP_dic['iy']
        self.line = FP_dic['i_line'] 
        self.sample = FP_dic['i_sample'] 
        self.LU = FP_dic['LU']
        self.satellite = FP_dic['satellite']
        self.grid = FP_dic['grid']
        self.gridName = FP_dic['gridName']
        self.timezone = FP_dic['timezone']    # can be UTC or solar local time
        self.LU_metadata = FP_dic['land_use_metadata']

        self.nFires = self.FRP.shape[0]
        self.timeStart = FP_dic['timeStart']
        try: 
            self.QA_flag = FP_dic['QA_flag']
        except: pass                           # backward compatibility, not None but undefined
        return self
    
    #========================================================================

    def to_dic(self, FP_dic):
        #
        # creates the object from dictionary that should have all vectors
        #
        return {'lon':self.lon, 'lat':self.lat, 'time':self.time, 'FRP':self.FRP,
                'dS':self.dS, 'dT':self.dy, 'sza':self.SolZenAng, 
                'ViewZenAng':self.ViewZenAng, 'T4':self.T4, 'T4b':self.T4b, 
                'T11':self.T11, 'T11b':self.T11b, 'TA':self.TA, 'ix':self.ix, 'iy':self.iy,
                'i_line':self.i_line, 'i_sample':self.i_sample, 'LU':self.LU,
                'satellite':self. satellite, 'grid':self.grid, 'gridName':self.gridName,
                'timeStart':self.timeStart, 'timezone':self.timezone, 'land_use_metadata':self.LU_metadata,
                'QA_flag':self.QA_flag}
    
    #========================================================================

    def from_nc(self, chNC4_FNm_or_handler, sensor=None):
        #
        # Reads the fire records from file, either opening it explicitly, or from a handler
        # of already opened file
        #
        if type(chNC4_FNm_or_handler) is str:
            fIn = nc4.Dataset(chNC4_FNm_or_handler,'r')
            fIn.set_auto_mask(False)            ## Never mask
        else:
            fIn = chNC4_FNm_or_handler
        #
        # Do we add the records from the nc with the same-QA records?
        #
        try: QA_in = fIn.QA_flag
        except: QA_in = None
 
        try:
            if self.QA_flag is None and QA_in is None: pass    # No QA in either of the files
            elif self.QA_flag != QA_in:      # new nc add-on to existing records
                raise ValueError('Input nc file has different QA: %i - then mine: %i' % 
                                 (fIn.QA_flag, self.QA_flag))
        except: self.QA_flag = QA_in     # this is the first nc
        #
        # Space?
        #
        try:
            if self.nFires+fIn.nFires > self.FRP.shape[0]:
                self.expand(max(self.nFires+fIn.nFires, int(self.nFires * 1.2)))
        except:
            self.init_data_structures(fIn.nFires)

        grid, gridName = silamfile.read_grid_from_nc(fIn)
        if self.grid is None: 
            self.grid = grid
            self.gridName = gridName
        else:
            if gridName != self.gridName:
                if self.gridName == 'undefined': 
                    self.log.log('Replacing my underfined gridName with:' + gridName)
                    self.gridName = gridName
                elif gridName == 'undefined': pass
                else: raise ValueError('gridName from external file is not the same as mine:\n mine:' + self.gridName + '\nTheirs:\n' + gridName)
            if grid != self.grid:
                raise ValueError('grid from external file is not the same as mine:\n mine:' + self.grid.toCDOgrid().tostr() + '\nTheirs:\n' + grid.toCDOgrid().tostr())
        # Land Use
        self.LU_metadata = fIn.land_use_metadata
        # Fires
        if fIn.nFires > 0:
            nF1 = self.nFires
            nF2 = self.nFires+fIn.nFires
            self.FRP[nF1:nF2] = fIn.variables['FP_frp'][:].astype(np.float32)[:fIn.nFires]
            self.lon[nF1:nF2] = fIn.variables['FP_lon'][:].astype(np.float32)[:fIn.nFires]
            self.lat[nF1:nF2] = fIn.variables['FP_lat'][:].astype(np.float32)[:fIn.nFires]
            self.dS [nF1:nF2] = fIn.variables['FP_dS'][:].astype(np.float32)[:fIn.nFires]
            self.dT [nF1:nF2] = fIn.variables['FP_dT'][:].astype(np.float32)[:fIn.nFires]
            self.T4 [nF1:nF2] = fIn.variables['FP_T4'][:].astype(np.float32)[:fIn.nFires]
            self.T4b[nF1:nF2] = fIn.variables['FP_T4b'][:].astype(np.float32)[:fIn.nFires]
            self.T11[nF1:nF2] = fIn.variables['FP_T11'][:].astype(np.float32)[:fIn.nFires]
            self.T11b[nF1:nF2]= fIn.variables['FP_T11b'][:].astype(np.float32)[:fIn.nFires]
            self.TA [nF1:nF2] = fIn.variables['FP_TA'][:].astype(np.float32)[:fIn.nFires]
            self.ix [nF1:nF2] = fIn.variables['FP_ix'][:].astype(np.int32)[:fIn.nFires]
            self.iy [nF1:nF2] = fIn.variables['FP_iy'][:].astype(np.int32)[:fIn.nFires]
            self.LU [nF1:nF2] = fIn.variables['FP_LU'][:].astype(np.int16)[:fIn.nFires]
            #
            # RELATIVE time is stored: the start time and a vector of seconds passed since then
            ## native fire_records or dailyMaps?
            if 'FP_time' in fIn.variables:   
                # native
                chTimeStep = fIn.variables['FP_time'].units.split()[0]
                timezone = fIn.FP_timezone
                timeStart = dt.datetime.strptime(fIn.variables['FP_time'].units,
                                                 chTimeStep +' since %Y-%m-%d %H:%M:%S ' + timezone)
                chTimeKey = 'FP_time'
                factor = 1
            else:
                # dailyMaps. Note that units may contain WRONG timezone. Use ifLocalTime from maps
                chTimeStep = fIn.variables['time'].units.split()[0]
                timezone = {True: 'LST', False: 'UTC'}[fIn.ifLocalTime.upper() == 'TRUE']
                timeStart = dt.datetime.strptime(fIn.variables['time'].units,
                                                 chTimeStep +' since %Y-%m-%d %H:%M:%S ' + 
                                                 fIn.variables['time'].units.split()[-1])
                chTimeKey = 'FP_hour'
                factor = 3600
            #
            # new file or appending?
            #
            if self.timeStart is None:   
                # empty file: set the start from here
                self.timeStart = timeStart
                deltaTime = 0
                self.timezone = timezone
            else:
                # data exist, set the right intervals
                if self.timezone != timezone: 
                    raise ValueError('Different timezones: %s vs %s' % (self.timezone, timezone))
                deltaTime = (timeStart - self.timeStart).total_seconds()
            # times themselves
            self.time[nF1:nF2] = fIn.variables[chTimeKey][:][:fIn.nFires].astype(np.int64) * factor + deltaTime
            self.chTimeStep = 'seconds'
            
            try:
                self.line[nF1:nF2] = fIn.variables['FP_line'][:].astype(np.float32)[:fIn.nFires]
                self.sample[nF1:nF2] = fIn.variables['FP_sample'][:].astype(np.float32)[:fIn.nFires]
            except:
                self.line[nF1:nF2] = np.zeros(shape=(fIn.nFires), dtype=np.float32)
                self.sample[nF1:nF2] = np.zeros(shape=(fIn.nFires), dtype=np.float32)

            try: 
                self.SolZenAng[nF1:nF2] = fIn.variables['FP_SolZenAng'][:].astype(np.float32)[:fIn.nFires]
            except:
                dt_time = self.timeStart + spp.one_minute * 60. * self.time[nF1:nF2]
                jday = np.array(list((t.timetuple().tm_yday for t in dt_time)))
                hours = np.array(list((t.hour for t in dt_time)))
                minutes = np.array(list((t.hour for t in dt_time)))
                # Turning minutes since specific data to needed variables
                self.SolZenAng[nF1:nF2] = spp.solar_zenith_angle(
                                    self.lon[:fIn.nFires], self.lat[:fIn.nFires], jday, hours, minutes)

            try: self.ViewZenAng[nF1:nF2] = fIn.variables['FP_ViewZenAng'][:].astype(np.float32)[:fIn.nFires]
            except: self.ViewZenAng[nF1:nF2
                                    ] = np.zeros(shape=(fIn.nFires), dtype=np.float32)

            try: self.satellite[nF1:nF2
                                ] = fIn.variables['FP_satellite'][:].astype('|S1')[:fIn.nFires]
            except:
                #
                # Historical issue: MODIS daily files do not contain satellite info but it can be 
                # restored from timing and position
                #
                if sensor is None:
                    self.satellite[nF1:nF2] = b'-'
                else:
                    self.restore_satellite(sensor, nF1, nF2)

            
            self.nFires += fIn.nFires
        return self


    #=======================================================================
    
    def collect_from_files(self, lstFiles):
        #
        # Goes through the bunch of files getting the fire records
        #
        # Prepare the space if needed
        #
        ifReinit = True
        try:
            if self.frp.shape[0] > 10000000: ifReinit = False
        except: pass
        if ifReinit: self.init_data_structures(10000000)
        #
        # Go through the files
        #
        for cnt, inF in enumerate(lstFiles):
            if np.mod(cnt,100) == 0: print(inF)
            self.from_nc(inF)

        return self
        

    #=======================================================================
    
    def to_nc(self, chOut_FNm_or_handle):
        #
        # Stores the produced fire records into the netCDF file. 
        #
        # file name or handler?
        if type(chOut_FNm_or_handle) is str:
            outF = nc4.Dataset(chOut_FNm_or_handle , "w", format="NETCDF4")
        else:
            outF = chOut_FNm_or_handle

        outF.featureType = "timeSeries";
        outF.FP_timezone = self.timezone
        outF.land_use_metadata = self.LU_metadata
        if self.grid is not None:
            silamfile.write_grid_to_nc(outF, self.grid, self.gridName) 
        outF.nFires = self.nFires
        # QA
        try: self.QA.to_nc(outF)
        except: pass
        outF.QA_flag = self.QA_flag
        #
        # The only dimension
        #
        firesAxis = outF.createDimension("fires", self.nFires)

        # variables
        # some versions of ncdump have issues parsing minutes in reftime but nothing to do...

        for FP_var in [('FP_frp','f4','FRP','MW', self.FRP[:self.nFires]),   # var_name, type, long name, unit
                       ('FP_time', "i4","time", 
                        self.timeStart.strftime("seconds since %Y-%m-%d %H:%M:%S " + self.timezone), 
#                        (self.time[:self.nFires] - self.time[0])/spp.one_minute),
                        self.time[:self.nFires]),
                       ('FP_lon','f4','longitude','degrees_east', self.lon[:self.nFires]),
                       ('FP_lat','f4','latitude','degrees_north', self.lat[:self.nFires]),
                       ('FP_dS','f4','pixel size along swath','km', self.dS[:self.nFires]),
                       ('FP_dT','f4','pixel size along trajectory','km', self.dT[:self.nFires]),
                       ('FP_T4','f4','temperature 3.96 um','K', self.T4[:self.nFires]),
                       ('FP_T4b','f4','temperature background 3.96 um','K', self.T4b[:self.nFires]),
                       ('FP_T11','f4','temperature 11 um','K', self.T11[:self.nFires]),
                       ('FP_T11b','f4','temperature background 11 um','K', self.T11b[:self.nFires]),
                       ('FP_TA','f4','temperature anomaly','K', self.TA[:self.nFires]),
                       ('FP_ix','i4','pixel x-index','', self.ix[:self.nFires]),
                       ('FP_iy','i4','pixel y-index','', self.iy[:self.nFires]),
                       ('FP_line','i4','granule line of fire pixel','', self.line[:self.nFires]),
                       ('FP_sample','i4','granule sample of fire pixel','', self.sample[:self.nFires]),
                       ('FP_SolZenAng','f4','solar zenith angle','degrees', self.SolZenAng[:self.nFires]),
                       ('FP_ViewZenAng','f4','view zenith angle','degrees', self.ViewZenAng[:self.nFires]),
                       ('FP_satellite','S1','satellite 1-letter name','', self.satellite[:self.nFires]),
                       ('FP_LU','i2','land use index','', self.LU[:self.nFires])]:
            vFP = outF.createVariable(FP_var[0], FP_var[1], ("fires"), zlib=True, complevel=5)
#                                          least_significant_digit=5)
            vFP.long_name = FP_var[2]
            if FP_var[3] != '': vFP.units = FP_var[3]
#            print(FP_var[0])
            outF.variables[FP_var[0]][:] = FP_var[4][:self.nFires]  # shapes of arrays can be larger

        if type(chOut_FNm_or_handle) is str:
            outF.close()

    #=======================================================================
    
    def sort(self):
        #
        # Makes a copy of the self with (i) cutting empty spaces up to nFires, (ii) sorting
        # the fire records along the time axis
        #
        FP_out = fire_records(self.log)  # start new object
        FP_out.nFires = self.nFires
        FP_out.timeStart = self.timeStart
        FP_out.timezone = self.timezone
        FP_out.LU_metadata = self.LU_metadata
        try: FP_out.chTimeStep = self.chTimeStep
        except:
            self.log.log('Force seconds') 
            FP_out.chTimeStep = 'seconds'      # for whatever reason, may be absent
        FP_out.grid = copy.copy(self.grid)
        FP_out.gridName = copy.copy(self.gridName)
        FP_out.QA_flag = self.QA_flag
        try: FP_out.QA = self.QA
        except: pass   # basically, if QA_flag is trivial, no need in QA 
#        FP_out.QA = self.QA
        #
        # Sorting index: follow time
        #
        idxSort = np.argsort(self.time[:self.nFires])
        
        FP_out.lon = self.lon[:self.nFires][idxSort]
        FP_out.lat = self.lat[:self.nFires][idxSort]
        FP_out.time = self.time[:self.nFires][idxSort]
        FP_out.FRP = self.FRP[:self.nFires][idxSort]
        FP_out.dS = self.dS[:self.nFires][idxSort]
        FP_out.dT = self.dT[:self.nFires][idxSort]
        FP_out.T4 = self.T4[:self.nFires][idxSort]
        FP_out.T4b = self.T4b[:self.nFires][idxSort]
        FP_out.T11 = self.T11[:self.nFires][idxSort]
        FP_out.T11b = self.T11b[:self.nFires][idxSort]
        FP_out.TA = self.TA[:self.nFires][idxSort]
        FP_out.ix = self.ix[:self.nFires][idxSort]
        FP_out.iy = self.iy[:self.nFires][idxSort]
        FP_out.LU = self.LU[:self.nFires][idxSort]
        FP_out.line = self.line[:self.nFires][idxSort]
        FP_out.sample = self.sample[:self.nFires][idxSort]
        FP_out.SolZenAng = self.SolZenAng[:self.nFires][idxSort]
        FP_out.ViewZenAng = self.ViewZenAng[:self.nFires][idxSort]
        FP_out.satellite = self.satellite[:self.nFires][idxSort]
        FP_out.sorted = True
        return FP_out

    #=======================================================================
    
    def expand(self, newSize):
        #
        # Adds new space
        #
        addOn = newSize - self.nFires
        if addOn < 0: raise ValueError('Cannot expand to a smaller size: %g to %g' % (self.nFires, newSize))
        print('Current and new sizes, addOn:', self.lon.shape[0], newSize, addOn)
        # may be, the self structures are undefined?
        ifInit = False
        try:
            ifInit = self.FRP.shape[0] == 0
        except: ifInit = True
        # Initialising or expanding
        if ifInit:
            print('Initializing...') 
            self.init_data_structures(newSize)
        else:
            self.lon = np.concatenate((self.lon, np.zeros(shape=(addOn),dtype=np.float32)))
            self.lat = np.concatenate((self.lat, np.zeros(shape=(addOn),dtype=np.float32)))
            self.time = np.concatenate((self.time, np.zeros(shape=(addOn),dtype=np.int64)))
            self.FRP = np.concatenate((self.FRP, np.zeros(shape=(addOn),dtype=np.float32)))
            self.dS = np.concatenate((self.dS, np.zeros(shape=(addOn),dtype=np.float32)))
            self.dT = np.concatenate((self.dT, np.zeros(shape=(addOn),dtype=np.float32)))
            self.T4 = np.concatenate((self.T4, np.zeros(shape=(addOn),dtype=np.float32)))
            self.T4b = np.concatenate((self.T4b, np.zeros(shape=(addOn),dtype=np.float32)))
            self.T11 = np.concatenate((self.T11, np.zeros(shape=(addOn),dtype=np.float32)))
            self.T11b = np.concatenate((self.T11b, np.zeros(shape=(addOn),dtype=np.float32)))
            self.TA = np.concatenate((self.TA, np.zeros(shape=(addOn),dtype=np.float32)))
            self.ix = np.concatenate((self.ix, np.zeros(shape=(addOn),dtype=int)))
            self.iy = np.concatenate((self.iy, np.zeros(shape=(addOn),dtype=int)))
            self.LU = np.concatenate((self.LU, np.zeros(shape=(addOn),dtype=int)))
            self.line = np.concatenate((self.line, np.zeros(shape=(addOn),dtype=int)))
            self.sample = np.concatenate((self.sample, np.zeros(shape=(addOn),dtype=int))) 
            self.SolZenAng = np.concatenate((self.SolZenAng, np.zeros(shape=(addOn),dtype=np.float32)))
            self.ViewZenAng = np.concatenate((self.ViewZenAng, np.zeros(shape=(addOn),dtype=np.float32)))
            self.satellite = np.concatenate((self.satellite, np.zeros(shape=(addOn),dtype='|S1')))


    #=======================================================================
    
    def append(self, FP_new, newSize=None):
        #
        # Appends FP_new fire records to the existing ones
        #
        if FP_new.nFires == 0: return self  # stupidity precaution: empty files are possible
        if FP_new.QA_flag != self.QA_flag: raise ValueError('Different QA_flag in new records: %i, from mine: %i' %
                                                            (FP_new.QA_flag, self.QA_flag))
        #
        # new size:
        # The simplest is just a sum of the two record lengths, but the driver can ask for a 
        # larger reserve to avoid too many concatenations 
        #
        if newSize is None: newSize = self.nFires + FP_new.nFires  # simple case
        newSize = max(self.nFires + FP_new.nFires, newSize)   # At least this
        # need more space?
        if self.lon.shape[0] < newSize: self.expand(newSize)
        #
        # copy the new data to the empty (possibly, just newly-created) space
        #
        iEnd = self.nFires + FP_new.nFires
        idxSort = np.argsort(FP_new.time[:FP_new.nFires])  # manual sorting, no need to create a sorted object
        self.lon[self.nFires:iEnd] = FP_new.lon[:FP_new.nFires][idxSort]
        self.lat[self.nFires:iEnd] = FP_new.lat[:FP_new.nFires][idxSort]
        self.FRP[self.nFires:iEnd] = FP_new.FRP[:FP_new.nFires][idxSort]
        self.dS[self.nFires:iEnd] = FP_new.dS[:FP_new.nFires][idxSort]
        self.dT[self.nFires:iEnd] = FP_new.dT[:FP_new.nFires][idxSort]
        self.T4[self.nFires:iEnd] = FP_new.T4[:FP_new.nFires][idxSort]
        self.T4b[self.nFires:iEnd] = FP_new.T4b[:FP_new.nFires][idxSort]
        self.T11[self.nFires:iEnd] = FP_new.T11[:FP_new.nFires][idxSort]
        self.T11b[self.nFires:iEnd] = FP_new.T11b[:FP_new.nFires][idxSort]
        self.TA[self.nFires:iEnd] = FP_new.TA[:FP_new.nFires][idxSort]
        self.ix[self.nFires:iEnd] = FP_new.ix[:FP_new.nFires][idxSort]
        self.iy[self.nFires:iEnd] = FP_new.iy[:FP_new.nFires][idxSort]
        self.LU[self.nFires:iEnd] = FP_new.LU[:FP_new.nFires][idxSort]
        self.line[self.nFires:iEnd] = FP_new.line[:FP_new.nFires][idxSort]
        self.sample[self.nFires:iEnd] = FP_new.sample[:FP_new.nFires][idxSort]
        self.SolZenAng[self.nFires:iEnd] = FP_new.SolZenAng[:FP_new.nFires][idxSort]
        self.ViewZenAng[self.nFires:iEnd] = FP_new.ViewZenAng[:FP_new.nFires][idxSort]
        self.satellite[self.nFires:iEnd] = FP_new.satellite[:FP_new.nFires][idxSort]
        # FP_new may have different starting time, adjust to self
        timeDelta = (self.timeStart - FP_new.timeStart).total_seconds()
        self.time[self.nFires:iEnd] = FP_new.time[:FP_new.nFires][idxSort] - timeDelta

        self.nFires += FP_new.nFires

        return self


    #=======================================================================
    
    def subset_time(self, dateStartNew, dateEndNew):
        #
        # Extracts a subset of fire records for the specific interval
        #
        FP_out = fire_records(self.log)  # start new object
        FP_out.timezone = self.timezone
        FP_out.LU_metadata = self.LU_metadata
        FP_out.grid = copy.copy(self.grid)
        FP_out.gridName = copy.copy(self.gridName)
        try: FP_out.QA = self.QA
        except: pass   # basically, if QA_flag is trivial, no need in QA 
        FP_out.QA_flag = self.QA_flag
        
        #
        # Sorting index: follow time
        #
        idxSort = np.argsort(self.time[:self.nFires])   # seconds since the timeStart
        secStart  = (dateStartNew - self.timeStart).total_seconds()
        secEnd = (dateEndNew - self.timeStart).total_seconds()
        iStart, iEnd = np.searchsorted(self.time[:self.nFires][idxSort], [secStart, secEnd]) 
        
        FP_out.lon = self.lon[:self.nFires][idxSort][iStart:iEnd]
        FP_out.lat = self.lat[:self.nFires][idxSort][iStart:iEnd]
        FP_out.time = self.time[:self.nFires][idxSort][iStart:iEnd]
        FP_out.FRP = self.FRP[:self.nFires][idxSort][iStart:iEnd]
        FP_out.dS = self.dS[:self.nFires][idxSort][iStart:iEnd]
        FP_out.dT = self.dT[:self.nFires][idxSort][iStart:iEnd]
        FP_out.T4 = self.T4[:self.nFires][idxSort][iStart:iEnd]
        FP_out.T4b = self.T4b[:self.nFires][idxSort][iStart:iEnd]
        FP_out.T11 = self.T11[:self.nFires][idxSort][iStart:iEnd]
        FP_out.T11b = self.T11b[:self.nFires][idxSort][iStart:iEnd]
        FP_out.TA = self.TA[:self.nFires][idxSort][iStart:iEnd]
        FP_out.ix = self.ix[:self.nFires][idxSort][iStart:iEnd]
        FP_out.iy = self.iy[:self.nFires][idxSort][iStart:iEnd]
        FP_out.LU = self.LU[:self.nFires][idxSort][iStart:iEnd]
        FP_out.line = self.line[:self.nFires][idxSort][iStart:iEnd]
        FP_out.sample = self.sample[:self.nFires][idxSort][iStart:iEnd]
        FP_out.SolZenAng = self.SolZenAng[:self.nFires][idxSort][iStart:iEnd]
        FP_out.ViewZenAng = self.ViewZenAng[:self.nFires][idxSort][iStart:iEnd]
        FP_out.satellite = self.satellite[:self.nFires][idxSort][iStart:iEnd]
        FP_out.sorted = True
        FP_out.nFires = len(FP_out.lon)
        # FP_out may have different starting time
        FP_out.timeStart = dateStartNew
        timeDelta = (self.timeStart - FP_out.timeStart).total_seconds()
        FP_out.time = FP_out.time + timeDelta

        return FP_out


    #=======================================================================
    
    def subset_fires(self, if_good_fires):
        #
        # Picks a subset of fire records taking only "good_fires", whatever it means
        # Returns a new fire_record object
        #
        FP_out = fire_records(self.log)  # start new object
        FP_out.timezone = self.timezone
        FP_out.LU_metadata = self.LU_metadata
        FP_out.timeStart = self.timeStart
        FP_out.grid = copy.copy(self.grid)
        FP_out.gridName = copy.copy(self.gridName)
        try: FP_out.QA = self.QA
        except: pass   # basically, if QA_flag is trivial, no need in QA 
        FP_out.QA_flag = self.QA_flag
        #
        # Sorting index: follow time
        #
        FP_out.lon = self.lon[:self.nFires][if_good_fires]
        FP_out.lat = self.lat[:self.nFires][if_good_fires]
        FP_out.time = self.time[:self.nFires][if_good_fires]
        FP_out.FRP = self.FRP[:self.nFires][if_good_fires]
        FP_out.dS = self.dS[:self.nFires][if_good_fires]
        FP_out.dT = self.dT[:self.nFires][if_good_fires]
        FP_out.T4 = self.T4[:self.nFires][if_good_fires]
        FP_out.T4b = self.T4b[:self.nFires][if_good_fires]
        FP_out.T11 = self.T11[:self.nFires][if_good_fires]
        FP_out.T11b = self.T11b[:self.nFires][if_good_fires]
        FP_out.TA = self.TA[:self.nFires][if_good_fires]
        FP_out.ix = self.ix[:self.nFires][if_good_fires]
        FP_out.iy = self.iy[:self.nFires][if_good_fires]
        FP_out.LU = self.LU[:self.nFires][if_good_fires]
        FP_out.line = self.line[:self.nFires][if_good_fires]
        FP_out.sample = self.sample[:self.nFires][if_good_fires]
        FP_out.SolZenAng = self.SolZenAng[:self.nFires][if_good_fires]
        FP_out.ViewZenAng = self.ViewZenAng[:self.nFires][if_good_fires]
        FP_out.satellite = self.satellite[:self.nFires][if_good_fires]
        FP_out.sorted = self.sorted
        FP_out.nFires = len(FP_out.lon)

        return FP_out


