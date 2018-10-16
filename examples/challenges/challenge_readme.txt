In the somewhat near future, we will provide synthetic light curves with hidden exoplanets.
These will allow users of the module to test out different detection schemes.
Their difficulty represents added levels of realism, starting with artificially good data,
then moving on to fully realistic detection scenarios barely above the S/N limit.

Overview:
The easy datasets are as easy as it gets.  Little noise, a delta function star, and a bright exoplanet in a circular orbit.
Often, the flares are actually delta functions.
easy_1.npy
easy_2.npy

The medium datasets no longer guarantee a circular orbit, but have good S/N conditions.
They approach more realistic observing scenarios.
medium_1.npy
medium_2.npy
medium_3.npy
medium_4.npy

The hard datasets no longer use a delta function star--some flares may miss the exoplanet.
Additionally, there may be complicated orbital conditions.
hard_1.npy
hard_2.npy
hard_3.npy
hard_4.npy

The realistic datasets are brutal, including as much realism as we can generate, but always ensuring the S/N conditions
can be overcome, with appropriate data processing.
realistic_1.npy

The cruel datasets do not guarantee an exoplanet in the system, so you may just discover noise.
Or maybe the exoplanet has a variable albedo.  Or maybe there are several exoplanets.  Who knows, it's anything goes.
cruel_1.npy
