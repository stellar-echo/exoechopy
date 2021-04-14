import numpy as np
import os
import sys
from pathlib import Path
from lowest_detectable_echo import test_detecting_synthetic_echoes, find_lowest_echo


# Load long cadence data

print("Loading Data.......")

# longcadence = []
# with open("longCadence_in_paper.txt", "r") as f:
#    for line in f:
#        longcadence.append(line.strip())

# Load short cadence data
shortcadence = []
with open("shortCadence_in_paper.txt", "r") as g:
    for line in g:
        shortcadence.append(line.strip())

# Adjust formatting -- remove leading "kplr"
# long_cadence_condensed = []
# for star in longcadence:
#    long_cadence_condensed.append(star[4:])

short_cadence_condensed = []
for star in shortcadence:
    short_cadence_condensed.append(star[4:])
    
print("Data loaded.")

# Removing problematic stars for now, will re-examine individually when done
# problematic_lc = ["008711794", "009475552", "011136848"]

# for i in long_cadence_condensed:
#    if i in problematic_lc:
#        long_cadence_condensed.remove(i)

print("-----------------------------------")

# Now perform the lowest detectable echo search, writing results to files when done.
# print("Finding lowest detectable echo for long cadence stars.......")

"""
lc_results = []
for i, star in enumerate(long_cadence_condensed):
    print("Checking", star)
    star_name, lde, dme = find_lowest_echo(star, 3, "long")
    lc_results.append((star_name, lde, dme))
    print("Got result for star ", star, "number ", i, "of ", len(long_cadence_condensed), (star_name, lde, dme)) 

print("Found all lowest detectable echo strengths for long cadence stars.")
print("Writing to file......")

with open('lc_results.txt', 'w') as f:
    for item in lc_results:
        f.write("%s \n" % str(item))
    f.close()

print("Wrote long cadence results to file.")
"""
print("----------------------------------------")

print("Finding lowest detectable echo for short cadence stars........")

sc_results = []
for i, star in enumerate(short_cadence_condensed):
    print("Checking", star)
    star_name, lde, dme = find_lowest_echo(star, 3, "short")
    sc_results.append((star_name, lde, dme))
    print("Got result for star ", star, "number ", i, "of ", len(short_cadence_condensed), (star_name, lde, dme))

print("Done with short cadence. Writing to file......")

with open("sc_results.txt", "w") as f:
    for item in sc_results:
        f.write("%s \n" % str(item))
    f.close()

print("Wrote short cadence results to file.")

print("========= done! :) ==========")

