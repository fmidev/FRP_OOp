# FRP_OOp
Observation Operator for Fire Radiative Power satellite products

The current repository is a Python-3 implementation of Observation Operator for
Fire Radiative Power products of satellite instruments. Today, parameters
of the OOp allow its application to MODIS and VIIRS level-2 data.

Prerequisites:

1. The OOp scripts rely on SILAM Python-3 environment. One can get it here:
https://github.com/fmidev/SILAM_python_3_toolbox

2. FORTRAN-Python interface f2py3. The processing of MODIS granules is heavy, so it was
implemented in FORTRAN and requires f2py3 installed with an appropriate FORTRAN compiler.

The main program is FRP_OOp_driver.py, which has all paths to input and output data
