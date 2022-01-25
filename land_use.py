'''

Class land_use
Handles the IS4FIRES v.3.x land-use class
Its implementation relies on a NetCDF map with codes and metadatafile

Extracted as a stand-alone module from IS4FIRES v.3.0 on 7.1.2022

Functions:
-  init:    a driver that calls one of the two subroutines:
 - from_metadata_file:  reads the metadatafile and the main nc file whose name is in the metadata file
 - from_nc_file:    same as above but reverse order: nc file should have a metadatafile name
- update_from_file: Reset only None components. Basic LU is already available
- get_LU_4_fires:   returns LU types for a bunch of coordinates
- get_LU_types:     returns the availabe LU types
- get_LU_diurnal_mean_map: Reads and returns LU-weighted mean diurnal variation in each cell.
- to_nc_file:       stores the land use reference names 
- add_halogens:     supplementary temporary sub increasing the list of species with halogenated compunds
- to_emission_cocktail:  Writes a SILAM cocktail description file
- to_metadata_file: Writes back the metadata file
- getGrids_LU:      returns subsets of the given grid, which cover the corresponding LUs

@author: Mikhail Sofiev
'''

import numpy as np
from toolbox import namelist, silamfile, gridtools
from os import path
import netCDF4 as nc4


#################################################################################
#
# Land use class: land use names, diurnal variations, maps of locations
# Land use (LU) has three main identifiers:
# - LUname, such as EU_boreal_forest
# - LUnbr in the map of land uses e.g. 403 for EU_boreal_forest
# - LUidx in the list pf land uses. Will be used to address the 2D duirnal variation
#   numpy array (hour_in_day, LUidx), size (24, nLU)
#
#################################################################################

class land_use():

    #==================================================================

    def __init__(self, chMetadataFNm=None, nc_file=None, chMetedata_dir=None):
        if chMetadataFNm is not None:
            self.from_metadata_file(chMetadataFNm)
        elif nc_file is not None:
            self.from_nc_file(nc_file, chMetedata_dir)
        else:
            self.LUtypes = None
            self.LU_full_data = False

    
    #==================================================================
    
    def from_metadata_file(self, chMetadataFNm):
        self.metadataFNm = chMetadataFNm
        # Initialises the land use class, reads the map and related metadata
        fMeta = open(chMetadataFNm, 'r')   # metadata file
        nlMeta = namelist.Namelist.fromfile(fMeta)
        fMeta.close()
        # LU number to LU name dictionary
        # list of tuplas <number> <name>; several numbers collaps to one name
        self.luNbr2name = {}
        for lu in nlMeta.get('land_use'):
            self.luNbr2name[int(lu.split()[0])] = lu.split()[1]
        self.luNbr2name[-1] = 'LU_all'  # key in dictionary, not the last element of an array
        # Diurnal variation dictionary: LU names to variation
        # Read to intermediate dictionary
        diurnal = {}
        for idxLine in nlMeta.get('hour_in_day_index_total'):
            diurnal[idxLine.split()[0]] = list( (float(s) for s in idxLine.split()[1:]) )
        diurnal_per_fire = {}
        for idxLine in nlMeta.get('hour_in_day_index_per_fire'):
            diurnal_per_fire[idxLine.split()[0]] = list( (float(s) for s in idxLine.split()[1:]) )
        # Full list of LUs with diurnal variations, sorted alphabetically 
        self.LUtypes = np.sort(np.array(list(diurnal.keys())))
        # Make the main dictionary of diurnal variation for named LUs
        self.diurnal = np.zeros(shape=(24, self.LUtypes.shape[0])) * np.nan
        self.diurnal_per_fire = np.zeros(shape=(24, self.LUtypes.shape[0])) * np.nan
        self.LUnbr2idx = np.zeros(shape=(1+np.max(np.array(list(self.luNbr2name.keys())))),
                                  dtype=int) 
        for luNbr in self.luNbr2name.keys():
            if luNbr == -1: pass
#                self.LUnbr2idx[luNbr] = self.LUtypes.shape[0]
#                self.diurnal[:,self.LUnbr2idx[luNbr]] = diurnal['void']    # Diurnal varition for the LU-type
#                self.diurnal_per_fire[:,self.LUnbr2idx[luNbr]] = diurnal_per_fire['void']    # Diurnl varition for the LU-types
            else:
                self.LUnbr2idx[luNbr] = np.searchsorted(self.LUtypes, self.luNbr2name[luNbr])
                if self.LUtypes[self.LUnbr2idx[luNbr]] != self.luNbr2name[luNbr]:
                    print('Strange LU type', self.luNbr2name[luNbr],'. LU number ',luNbr)
                    raise ValueError
                self.diurnal[:,self.LUnbr2idx[luNbr]] = np.array(diurnal[self.luNbr2name[luNbr]])
                self.diurnal_per_fire[:,self.LUnbr2idx[luNbr]] = np.array(diurnal_per_fire
                                                                          [self.luNbr2name[luNbr]])
#            print(9, luNbr, self.diurnal[:,35], self.LUnbr2idx[luNbr], self.luNbr2name[luNbr])
#        print(10, luNbr, self.diurnal[:,35])
        
        # Get the emission factors
        # line format:
        # emission_factor = SA_grass OC fire_pm_spectrum 0.00210916 0.00421833 kg/MJ
        self.emisFactor = {}
        for line in nlMeta.get('emission_factor'):
            flds = line.split()
            try:           #     LU       subst      gas/aer    flamingEF       smoulderingEF   unit
                self.emisFactor[flds[0]][flds[1]] = (flds[2], float(flds[3]), float(flds[4]), flds[5])
            except:
                self.emisFactor[flds[0]] = {flds[1]: (flds[2], float(flds[3]), float(flds[4]), flds[5])}
        # and halogens - no longer needed, files are correct now
#        self.add_halogens()
        #
        # Get the main map.
        # for projection, I will need reader too
        self.chMapFNm = nlMeta.get_uniq('land_use_file').split()[-1]
        if '^' in self.chMapFNm: 
            self.chMapFNm = path.join(path.split(chMetadataFNm)[0], self.chMapFNm.replace('^','')) 
        self.LUreader = silamfile.SilamNCFile(self.chMapFNm).get_reader('lutype', mask_mode=False)
        self.LUmap = self.LUreader.read()
        self.LU_full_data = True


    #==================================================================
    
    def from_nc_file(self, nc_file, chMetadata_dir=None):
        if isinstance(nc_file, str): fIn = nc4.Dataset(nc_file,'r')
        else: fIn = nc_file
        # nc file contains path to and the name of the metadata file
        self.metadataFNm = fIn.land_use_metadata
        # And the key variable: LUtypes
        self.LUtypes = nc4.chartostring(fIn.variables['land_use_code'][:])
        fIn.close()
        # LUtypes has been set but not yet the LUmap etc.
        self.LU_full_data = False
        # Metadata directory may not be the same as when the file was written but the file must be.
        if not path.exists(self.metadataFNm):
            if chMetadata_dir is None: return   # leave LUmap etc uninitialised
            else: self.metadataFNm = path.join(chMetadata_dir, path.split(self.metadataFNm)[1])
        # Now, initialize form the metadata file as usual
        self.from_metadata_file(self.metadataFNm)
        self.LU_full_data = True


    #==================================================================

    def update_from_file(self, nc_file, chMetadata_dir=None):
        # Reset only None components. Basic LU is already available
        if isinstance(nc_file, str): fIn = nc4.Dataset(nc_file,'r')
        else: fIn = nc_file
        # nc file contains path to and the name of the metadata file
        if self.metadataFNm is None: self.metadataFNm = fIn.land_use_metadata
        # And the key variable: LUtypes
        if self.LUtypes is None: self.LUtypes = nc4.chartostring(fIn.variables['land_use_code'][:])
        fIn.close()
        # LUtypes has been set but LUmap etc might or might not be needed:
        # someone could have estiablished them earlier
        if self.LU_full_data: return 
        # Metadata directory may not be the same as when the file was written but the file must be.
        if not path.exists(self.metadataFNm):
            if chMetadata_dir is None: return   # leave LUmap etc uninitialised
            else: self.metadataFNm = path.join(chMetadata_dir, path.split(self.metadataFNm)[1])
        # Now, initialize form the metadata file as usual
        self.from_metadata_file(self.metadataFNm)
         

    #==================================================================

    def get_LU_4_fires(self, lons, lats):
        # returns a vector of land use types as indices in teh sorted self.LUtypes array
        return self.LUnbr2idx[self.LUmap[self.LUreader.indices(lons, lats)].astype(int)]


    #==================================================================

    def get_LU_types(self):
        # Just encapsulation of the key data
        return self.LUtypes


    #==================================================================

    def get_LU_diurnal_mean_map(self, chLU_weighted_diurnal_map):
        # Reads and returns LU-weighted mean diurnal variation in each cell.
        # The map is created offline in fire_processor.fires_vs_landuse(...)
        # I have to assume that I know what this file is about.
        #
        # Get the file
        fIn = nc4.Dataset(chLU_weighted_diurnal_map,'r')
        # and read the map and the list of ordered landuses coded in the map
        # The index of the LU in the list corresponds to the number used for this LU in the map
        try:
            return (fIn.variables['LU_weighted_diurnal_coef'][:,:,:].data, # 24, ny, nx
                    nc4.chartostring(fIn.variables['land_use_code'][:]))
        except:
            return (fIn.variables['LU_weighted_diurnal_coef'][:,:,:], 
                    nc4.chartostring(fIn.variables['land_use_code'][:]))
        

    #==================================================================

    def to_nc_file(self, outF):
        # stores the land use reference names linked to the LU indices of the FP_LU vector
        # dimensions
        strlen = 64
        LUstrlen = outF.createDimension("name_strlen", strlen)  # 64 is string length
        LUtypes = outF.createDimension('LU_types', len(self.LUtypes))  # number of land uses
        # variable
        LUname = outF.createVariable("land_use_code","c",("LU_types","name_strlen"), 
                                     zlib=True, complevel=4)
        LUname.long_name = "land use code"
        # Storing the LU names goes via intermediate array: faster
        charTmp = np.zeros(shape=(len(self.LUtypes), strlen), dtype='S1')
        for iLU, LU in enumerate(self.LUtypes):
            charTmp[iLU,:] = nc4.stringtoarr(LU, strlen, dtype='S')
        LUname[:,:] = charTmp[:,:]
        # metadata: file name
        outF.land_use_metadata = self.metadataFNm


    #==================================================================

    def add_halogens(self):
        #
        # Adds emission factors for CH3Cl and CH3Br using the ratio of these factors 
        # in Akagi (2011) to those of van Marle et al, (2017), GMD 10, 3329-3357.
        # Additionally, for AGRI (Crop Residue), PEAT, and TEMF the values from DEFO (Tropical forest) is used.
        # SAVA: Savanna, grassland, and shrubland fires         
        # BORF: Boreal forest fires
        # TEMF: Temperate forest fires                       
        # DEFO: Tropical deforestation & degradation         
        # PEAT: Peat fires                                 
        # AGRI: Agricultural waste burning
        #
        # Reference data
        #
        EF_refs = {'CO'   :{'AGRI':102.,   'BORF':127.,   'DEFO':93.,    'PEAT':210.,   'SAVA':63.,     'TEMF':88.},
                   'CH3Cl':{'AGRI':0.05300,'BORF':0.05900,'DEFO':0.05300,'PEAT':0.05300,'SAVA':0.055000,'TEMF':0.05300},
                   'CH3Br':{'AGRI':0.00283,'BORF':0.00364,'DEFO':0.00283,'PEAT':0.00283,'SAVA':0.000853,'TEMF':0.00283},
                   'FIRE_PM2_5':{'AGRI':3.21 ,'BORF':5.2,   'DEFO':3.86,   'PEAT':3.04   ,'SAVA':4.17    ,'TEMF':2.8}}
        ratios_CO = {}
        for halo in ['CH3Cl','CH3Br']:
            ratios_CO[halo] = {}
            for LU in ['AGRI','BORF','DEFO','PEAT','SAVA','TEMF']:
                ratios_CO[halo][LU] = EF_refs[halo][LU] / EF_refs['CO'][LU]
        LU_loc = {'boreal_forest':'BORF', 'crop_residue':'AGRI', 'grass':'SAVA', 'nshrub':'SAVA', 'shrub':'SAVA',
                  'temperate_forest':'TEMF', 'tropical_forest':'DEFO'}
        #
        # Adding, same values for all continents
        #
        for LUbasic in 'boreal_forest crop_residue grass nshrub shrub temperate_forest tropical_forest'.split():
            for continent in 'AF AS ASE AU EU NA SA'.split():
                LU = continent + '_' + LUbasic
                for halo in ['CH3Cl','CH3Br']:
                    self.emisFactor[LU][halo] = ('gas',
                                                 ratios_CO[halo][LU_loc[LUbasic]] * self.emisFactor[LU]['CO'][1], #flaming
                                                 ratios_CO[halo][LU_loc[LUbasic]] * self.emisFactor[LU]['CO'][2], # smouldering
                                                 'kg/MJ')
            self.emisFactor['void']['CH3Cl'] = ('gas',0,0,'kg/MJ')
            self.emisFactor['void']['CH3Br'] = ('gas',0,0,'kg/MJ')
 

    #==================================================================

    def to_emission_cocktail(self, chCocktailNm):
        #
        # Writes a SILAM cocktail description file
        # Cocktails are made for each landuse, separately for flaming and smoundering 
        #
        # The fire aerosol spectrum is a prescribed set of two bins here:
        # aerosol_mode = 1  0.01   1  0.15  mkm   1100  kg/m3 
        # aerosol_mode = 2  1   10.   2.25  mkm   1100  kg/m3
        # I also prescribe here the fractionation: 0.75 - 0.25 for flaming
        #                                          0.72 - 0.28 for smouldering
        # See Notebook 12 pp.12-15 where these numbers are derived from 
        # Virkkula et al ACP, 2014
        #
        aerFract = {'flame':(0.75, 0.25), 'smould': (0.72, 0.28)}
        fOut = open(chCocktailNm,'w')
        for lu in self.emisFactor.keys():
            for iFlame, flame in enumerate(list(aerFract.keys())):
                fOut.write('COCKTAIL_DESCRIPTION_V3_2\n')
                fOut.write('cocktail_name = FIRE_%s_%s\n' % (lu, flame))
                fOut.write('mass_unit = kg\n')
                fOut.write('gas_phase = YES\n')
                fOut.write('if_normalise  = NO\n')  # to allow the sum of fractions != 1
                fOut.write('aerosol_mode = 1  0.01  1.  0.15  mkm   1100  kg/m3\n') 
                fOut.write('aerosol_mode = 2  1.   10.  2.25  mkm   1100  kg/m3\n') 
                fOut.write('mode_distribution_type = FIXED_DIAMETER\n')  
                for species in self.emisFactor[lu].keys():
                    if self.emisFactor[lu][species][0].upper() == 'GAS':
                        fOut.write('component_fraction = %s 0. 0. %g\n' % 
                                   (species, self.emisFactor[lu][species][iFlame+1]))
                    else:
                        fOut.write('component_fraction = %s %g %g 0.\n' % 
                                   (species, 
                                    self.emisFactor[lu][species][iFlame+1] * aerFract[flame][0],
                                    self.emisFactor[lu][species][iFlame+1] * aerFract[flame][1]))
                fOut.write('END_COCKTAIL_DESCRIPTION\n\n')
        fOut.close() 


    #==================================================================

    def to_metadata_file(self, chMetadataOutFNm):
        #
        # Writes back the metadata file
        #
        # First, copy the introduction
        #
        fIn = open(self.metadataFNm, 'r')
        fOut = open(chMetadataOutFNm,'w')
        fOut.write('#### Regenerated metadata file from %s\n' % self.metadataFNm)
        fOut.write('# The following lines have been copied from it\n')
        for line in fIn:
            if line.strip().startswith('#') or line.strip().startswith('!'):
                fOut.write(line)
                continue
            break
        # Now, the metadata themselves
        fOut.write('\nFIRE_METADATA_V1\n\n')
        fOut.write('  land_use_file = NETCDF:LandUseType ^%s\n' % self.chMapFNm)
        fOut.write('  fire_aerosol_spectrum = LOGNORMAL_THREE_MODES\n')
        fOut.write('  fraction_of_stem_mass = 0.1\n\n')
        # Emission factors
        for LU in self.LUtypes:
            for subst in self.emisFactor[LU].keys():
                fOut.write('emission_factor = %s %s %s %g %g %s\n' % 
                           (LU, subst, self.emisFactor[LU][subst][0], self.emisFactor[LU][subst][1],
                            self.emisFactor[LU][subst][2], self.emisFactor[LU][subst][3]))
        # Diurnal variation   hour_in_day_index_total 
        for iLU, LU in enumerate(self.LUtypes):
            fOut.write('  hour_in_day_index_total = %s %s\n' %
                       (LU, ' '.join('%g' %v for v in self.diurnal[:,iLU])))
        fOut.write('\n\n')
        for iLU, LU in enumerate(self.LUtypes):
            fOut.write('  hour_in_day_index_per_fire = %s %s\n' %
                       (LU, ' '.join('%g' %v for v in self.diurnal_per_fire[:,iLU])))
        fOut.write('\n\n')
        # LU coding
        for iCode in self.luNbr2name.keys():
            if iCode >= 0:
                fOut.write('land_use = %g %s\n' % (iCode, self.luNbr2name[iCode]))
        fOut.write('END_FIRE_METADATA_V1\n')         
        fOut.close()


    #================================================================
    
    def getGrids_LU(self, gridIn):
        #
        # LUs are limited in area: continents at the very least, also big continents have LU
        # covering only part of those 
        # This function returns subsets of the given grid, which cover the corresponding LUs
        
        
        # self.LUnbr2idx = np.zeros(shape=(1+np.max(np.array(list(self.luNbr2name.keys()))))
        #
        ix_minmax = np.zeros(shape=(len(self.LUtypes),2), dtype=np.int16)  # lons covering the LU
        iy_minmax = np.zeros(shape=(len(self.LUtypes),2), dtype=np.int16)  # lats covering the LU
        map_iLU = np.zeros(shape=self.LUmap.shape)
        dicLUgrids = []
        #
        # re-encode the LU map into the LU types indices
        for i in range(np.min(self.LUmap).astype(np.int16), np.max(self.LUmap).astype(np.int16)+1):
            if np.mod(i,100) == 0: print(i)
            map_iLU[self.LUmap == i] = self.LUnbr2idx[i]
        #
        # get the borders: calculate indices in the input grid, then take thjeir min and max
        for iLU, chLU in enumerate(self.LUtypes):
            idxLU = np.where(map_iLU == iLU)
            if idxLU[0].size == 0:
                dicLUgrids.append(gridtools.Grid(gridIn.x0, gridIn.dx, 1,
                                                 gridIn.y0, gridIn.dy, 1,
                                                 gridIn.proj))
                continue
            lons, lats = self.LUreader.grid.grid_to_geo(idxLU[0], idxLU[1])
            fXOut, fYOut = gridIn.geo_to_grid(lons, lats)
            # min-max box for LU indices
            ix_minmax[iLU,:] = [np.maximum(0, round(np.min(fXOut))-2),
                                np.minimum(gridIn.nx, round(np.max(fXOut))+2)]
            iy_minmax[iLU,:] = [np.maximum(0, round(np.min(fYOut))-2),
                                np.minimum(gridIn.ny, round(np.max(fYOut))+2)]
            # define the grid covering this LU
            dicLUgrids.append(gridtools.Grid(gridIn.x0 + ix_minmax[iLU,0] * gridIn.dx, 
                                             gridIn.dx, 
                                             ix_minmax[iLU,1] - ix_minmax[iLU,0],   #gridIn.nx, 
                                             gridIn.y0 + iy_minmax[iLU,0] * gridIn.dy,
                                             gridIn.dy, 
                                             iy_minmax[iLU,1] - iy_minmax[iLU,0],   #gridIn.ny, 
                                             gridIn.proj))
        return (dicLUgrids, ix_minmax, iy_minmax)

