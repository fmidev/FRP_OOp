'''
This is the module for basic VIIRS fire processing

Created on 31.10.2021

@author: sofievm
'''

import glob, os
import csv
import datetime as dt
import pickle
import numpy as np
import process_fires_gen as fpg
from toolbox import supplementary as spp
from support import suntime 


#########################################################################################

class VIIRS_fire_list():
    #
    # Class for holding the VIIRS fire data as obtained from FIRMS fire lists
    #
    def __init__(self, log):
        self.accept_confidence = ['n','h']  # 'l' = low is excluded
        self.accept_fire_type = ['0']  # only vegetation fire; 1=volcano, 2=static land source, 3=water
        self.dates = None
        self.log = log
    
    #=======================================================================
    
    def read_FIRM_csv(self, chFNm):
        #
        # Reads the CSV file as made by FIRMS distribution channel
        #
        with open (chFNm) as fIn:
            rdr = csv.DictReader(fIn)
            cnt = 0
            for line in rdr:
#                if cnt > 10000: break
                if line['type'] in self.accept_fire_type:
                    if line['confidence'] in self.accept_confidence:
                        # stupidity check
                        frp = float(line['frp'])
                        if frp > 1000:
                            print(line)
                            continue
                        # A new day?
#                        day = dt.datetime.strptime(line['acq_date'] + '_' + 
#                                                   line['acq_time'], '%Y-%m-%d_%H%M')
                        day = dt.datetime.strptime(line['acq_date'], '%Y-%m-%d')
                        date = day.date()
                        if self.dates is None:  # first time in the cycle 
                            self.dates = [date]
                            self.frp = [frp]
                            self.lon = [float(line['longitude'])]
                            self.lat = [float(line['latitude'])]
                            self.dx = [float(line['track'])]
                            self.dy = [float(line['scan'])]
                            self.time = [int(line['acq_time'])]
                            self.T4 = [float(line['bright_ti4'])]
                            self.T5 = [float(line['bright_ti5'])]
                            self.satellite = [line['satellite']]
                            continue
#                        elif date == self.dates[-1]:  # the most-frequent case
#                            pass
#                        elif date < self.dates[-1]:   # impossible!
#                            self.log.log('Wrong date order. New date %s, last date in list %s' %
#                                         (self.dates[-1].strftime('%Y-%m-%d'),
#                                          date.strftime('%Y-%m-%d')))
#                            raise ValueError
#                        elif date > self.dates[-1]:   # New day
#                            self.dates.append(date)
#                            self.frp.append([float(line['frp'])]])
#                            self.lon.append([[float(line['longitude'])]])
#                            self.lat.append([[float(line['latitude'])]])
#                            self.dx.append([[float(line['track'])]])
#                            self.dy.append([[float(line['scan'])]])
#                            self.time.append([[int(line['acq_time'])]])
#                            self.T4.append([[float(line['bright_ti4'])]])
#                            self.T5.append([[float(line['bright_ti5'])]])
#                            self.satellite.append([[line['satellite']]])
#                            continue
                        # add the new fire to the last day
                        self.frp.append(frp)
                        self.lon.append(float(line['longitude']))
                        self.lat.append(float(line['latitude']))
                        self.dx.append(float(line['track']))
                        self.dy.append(float(line['scan']))
                        self.dates.append(date)
                        self.time.append(int(line['acq_time']))
                        self.T4.append(float(line['bright_ti4']))
                        self.T5.append(float(line['bright_ti5']))
                        self.satellite.append(line['satellite'])
                cnt += 1
        print('File %s consumed. Total nbr of fires = %g' % (chFNm, len(self.frp)))

    #======================================================================

    def to_numpy_array(self):
        #
        # A technicality: turn all lists to numpy arrays
        #
        self.frp = np.array(self.frp, dtype=np.float32)
        self.lon =  np.array(self.lon, dtype=np.float32)
        self.lat =  np.array(self.lat, dtype=np.float32)
        self.dx=  np.array(self.dx, dtype=np.float32)
        self.dy=  np.array(self.dy, dtype=np.float32)
        self.dates =  np.array(self.dates)
        self.time =  np.array(self.time)
        self.T4 =  np.array(self.T4, dtype=np.float32)
        self.T5 =  np.array(self.T5, dtype=np.float32)
        self.satellite =  np.array(self.satellite)
        


#########################################################################################

def make_histogram(lstCSV_FNms, chOutDir, ifUsePickle, log):
    #
    # Reads the bunch of given files and calculates the histogram of the consumed fires
    #
    print('Reading the fires\n', '\n'.join(lstCSV_FNms),'\n')
    sizeLims = (0.,1.0)
    frpLims = [(0,1000),(0,10),(0,50),(0,75),(50,500),(500,5000),(0,5000)]
    FRPs = None
    size = None
    #
    # Processing should go year by year: too many fires to pick at once
    #
    for chFNm in lstCSV_FNms:
        # Read the csv-s or pickle
        if ifUsePickle and os.path.exists(chFNm + '.pickle'):
            with open(chFNm + '.pickle', 'rb') as handle:
                frpVIIRS = pickle.load(handle)
                print('Pickle %s consumed. Total nbr of fires = %g, max FRP = %g' % 
                      (chFNm, len(frpVIIRS.frp), np.max(frpVIIRS.frp)))
        else:
            frpVIIRS = VIIRS_fire_list(log)
            frpVIIRS.read_FIRM_csv(chFNm)
            frpVIIRS.to_numpy_array()
            if ifUsePickle: 
                with open(chFNm + '.pickle', 'wb') as handle:
                    pickle.dump(frpVIIRS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # Accumulate the needed parameters
        #
        idxOK = frpVIIRS.frp < 100000
        if FRPs is None:
            FRPs = frpVIIRS.frp[idxOK].copy()
            lons = frpVIIRS.lon[idxOK].copy()
            lats = frpVIIRS.lat[idxOK].copy()
            dates = frpVIIRS.dates[idxOK].copy()
            FRP_times = frpVIIRS.time[idxOK].copy()
            dxs = frpVIIRS.dx[idxOK].copy()
            dys = frpVIIRS.dy[idxOK].copy()
            szas = spp.solar_zenith_angle(frpVIIRS.lon[idxOK], frpVIIRS.lat[idxOK], 
                                          np.array(list((d.timetuple().tm_yday for d in frpVIIRS.dates[idxOK]))),
                                          (frpVIIRS.time[idxOK]/100).astype(np.int16),
                                          np.mod(frpVIIRS.time[idxOK],100))
        else:
            FRPs = np.concatenate((FRPs, frpVIIRS.frp[idxOK]), axis=0)
            lons = np.concatenate((lons,frpVIIRS.lon[idxOK]), axis=0)
            lats = np.concatenate((lats,frpVIIRS.lat[idxOK]), axis=0)
            dates = np.concatenate((dates,frpVIIRS.dates[idxOK]), axis=0)
            FRP_times = np.concatenate((FRP_times,frpVIIRS.time[idxOK]), axis=0)
            dxs = np.concatenate((dxs,frpVIIRS.dx[idxOK]))
            dys = np.concatenate((dys,frpVIIRS.dy[idxOK]))
            tmp = spp.solar_zenith_angle(frpVIIRS.lon[idxOK], frpVIIRS.lat[idxOK], 
                                         np.array(list((d.timetuple().tm_yday for d in frpVIIRS.dates[idxOK]))),
                                         (frpVIIRS.time[idxOK]/100).astype(np.int16),
                                         np.mod(frpVIIRS.time[idxOK],100))
            szas = np.concatenate((szas,tmp))
    #
    # The input is prepared. Reshape: separate days as a dimension
    #
    lstDates = sorted(list(set(dates)))
    #
    # Make and draw the histogram
    #
    fire_records = (lons, lats, dates, FRPs, FRP_times, dxs, dys, szas)
    fpg.fires_vs_pixel_size(fire_records, os.path.join(chOutDir,'VIIRS_histogr'), 
                            'all', True, sizeLims=(0.,1.0), log=log)


#########################################################################################

if __name__ == '__main__':
    print('Hi')

    ifHistogram = True
    
    chDataDir = 'f:\\data\\VIIRS_active_fires'
    chDataFNmTempl = 'fire_archive_V*.csv'
    chOutDir = 'f:\\project\\fires\\VIIRS'
    
    spp.ensure_directory_MPI(chOutDir, ifPatience=False)
    
    if ifHistogram:
        make_histogram(glob.glob(os.path.join(chDataDir, chDataFNmTempl)),
                       chOutDir, True,
                       spp.log(os.path.join(chOutDir,'FRP_distr.log')))


