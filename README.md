# FRP_OOp
Observation Operator for Fire Radiative Power satellite products

The current repository is a Python-3 implementation of Observation Operator for
Fire Radiative Power products of satellite instruments. Today, parameters
of the OOp allow its application to MODIS and VIIRS level-2 data.

Pre-requisites:
1. The OOp scripts rely on SILAM Python-3 environment. One can get it here:
https://github.com/fmidev/SILAM_python_3_toolbox

Known issue:
Reader for MODIS HDF files is known to fail in some computers. Should this happens, 
one can try to use an HDF for Python library pyhdf, which will become the main reader 
of this OOp in near future.

