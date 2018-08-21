# exoechopy

# Basics of ExoEchoPy

## Stellar Echo Detection
Stellar echo detection is an experimental method of detecting exoplanets. Theoretically, one should be able to detect the echo of a stellar flare off the surface of an exoplanet through its lightcurve. Realistically, this requires advanced data analysis tools. EchoPy is a simulation and modeling module that is designed to help develop those tools by generating synthetic data of a flare star and its exoplanets.

## Generating Data with ExoEchoPy

EchoPy generates lightcurves of a star with an orbiting exoplanet. The idea is to allow the user to search for the echo of the star's flares off the surface of the exoplanet. With EchoPy, the user will be able to specify a large array of parameters for the star-exoplanet system, as well as for the instruments used to observe the system. They will then be able to test various methods of detecting these echoes on the data.

--

## How to Install ExoEchoPy

In order to use ExoEchoPy as a module, the user must go to the directory that contains the repository exoechopy and type the following:

'''pip install -e exoechopy'''

This will allow the user to import ExoEchoPy as a module and to use the various data generation scripts within EchoPy.

--

## How to Generate Data

Data generation is done using start_observations.py, located in exoechopy/generate_data. In start_observations.py, you can edit observational parameters that specify details about the star system. Once you run start_observations.py, the workflow is as follows:

1. Read in specified parameters
2. Creates directory in specified location and saves the parameters to a config file
3. Runs generate_light_curve.py, which reads in the .config file, generates and saves the data, and saves plots of flare and echo statistics

Everything created by generate_light_curve.py is saved to the path that is specified by the "folder" parameter in start_observations.py.

**NOTE:** It is possible to use generate_light_curve.py if you already have a config file. The syntax is as follows:

'''python generate_light_curve.py [path to config file]'''

### Parameter Information

##### Units

All units are listed next to the parameter and all angles are in radians.

##### Generate light curves or flares only

ExoEchoPy offers the option to generate the entire light curve as a whole, as though you were continuously observing the object for the duration of the run, or to generate the flares alone, as thought you already went through the data and picked out the flares and their echoes. The option to do this is specified by the "save_as" parameter.

##### Generating data for multiple star-exoplanet systems

In order to generate multiple light curves for systems with different stellar/exoplanet parameters, make the parameter a list. For example, to generate two light curves with stars of masses 0.5 M_sun and 3 M_sun, set the stellar mass parameter as a list containing these two values.

Multiple parameters can be changed for the stars and exoplanets. For example, you can specify two different stars by specifying two stellar masses and two stellar radii. EchoPy reads in the first element of each list and sets those as parameters belonging to "Star 1" and reads in the second element in each list and set those as parameters for "Star 2". EchoPy will read the stekkar parameters in as "Star 1", "Star 2", "Star 3", etc and it will read in the exoplanet parameters as "Exoplanet 1", "Exoplanet 2", "Exoplanet 3", etc. It will then generate light curves for all possible combinations of stars and exoplanets. For example, if you list 2 stellar masses and stellar radii and list 2 exoplanet masses, 4 light curves will be generated. All stellar parameters that are lists must be the same length and all exoplanet parameters that are lists must be the same length, but those two lengths can be different. You can describe 2 different stars and 3 different exoplanets.

The naming convention for the files with multiple parameters listed will be the name of the file you specified in the parameters, followed by " _star1", "_star2", etc or "_planet1", and so on. If you list multiple star and exoplanet parameters, the specified file name will be followed by "_star1_planet1", "_star1_planet2", "star2_planet1", etc.
