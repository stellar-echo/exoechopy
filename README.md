# ExoEchoPy

# Basics of ExoEchoPy
ExoEchoPy is a library for simulating a theoretical exoplanet detection method known as stellar echo detection.
It provides methods for simulation, detection, analysis, evaluation, and validation.
The library is primarily built on astropy, numpy, scipy, and matplotlib.

## Stellar Echo Detection
Stellar echo detection is an experimental method of detecting exoplanets. 
Theoretically, one should be able to detect the echo of a stellar flare off the surface of an exoplanet through its lightcurve. 
Realistically, this requires specific observing conditions, quasi-well-behaved targets, and advanced data analysis tools. 
ExoEchoPy provides simulation and modeling tools to help identify observation requirements and develop detection methods.
The \simulate submodule helps generate synthetic data of a flare star and its exoplanets, the \analyze submodule helps evaluate the results, the \experiment submodule will provide tools for performing parametric studies, and the library also contains a variety analytical methods, demos, and examples.
Much of this effort is motivated by recent developments described in our upcoming/recently published work: https://arxiv.org/abs/1808.07029
Also check out http://www.stellarecho.com/

## Generating Data with ExoEchoPy
ExoEchoPy generates light curves of a star with orbiting exoplanets. 
The idea is to allow the user to search for the echo of the star's flares off the surface of the exoplanet. 
With ExoEchoPy, the user will be able to specify a large array of parameters for the star-exoplanet system, as well as for the instruments used to observe the system. 
They will then be able to test various methods of detecting these echoes on the data.

## How to Install ExoEchoPy
pip install git+https://github.com/chrisma2n/exoechopy.git

## How to Generate Data
For now, see the demos in \examples.
The scripts labeled 'tutorial' walk through various demos in a reasonable order.

### Parameter Information

##### Units

Units are handled by astropy. 
Many functions, particularly class-internal functions, are handled in a unitless system.
These functions, and many associated variables, are designated _lw (to indicate 'lightweight') in their name.
Occasionally, functions and classes can be initialized without explicit units: this throws a warning and the default units will be described to the user.

##### Generate light curves or flares only

ExoEchoPy offers the option to generate the entire light curve as a whole, as though you were continuously observing the object for the duration of the run, or to generate the flares alone, as though you already went through the data and picked out the flares and their echoes.


##### Generating data for multiple star-exoplanet systems

To run multiple parameters and compare their results, use the \experiment package.  It's not built yet.


# Acknowledgements
This work is funded by NASA Grant 80NSSC18K0041 and is currently maintained by Nanohmics, Inc.