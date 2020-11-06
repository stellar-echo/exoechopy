import os
import sys
from pathlib import Path

if len(sys.argv) != 2:
   print(f'usage {argv[0]} star_id')
   quit()


star=sys.argv[1]

kpath = Path('/home/echo/hdd6tb/02_kepler_time_series_scripts/')

sc_qtr_list = []
lc_qtr_list = []

all_folders = os.listdir(kpath)
quarter_folders = [x for x in all_folders if '_Q' in x]
print(quarter_folders)
print()
for qd in quarter_folders:
   for f in os.listdir(kpath/qd):
      if star in f:
         if 'llc' in f:
            lc_qtr_list.append(kpath/qd/f)
         elif 'slc' in f:
            sc_qtr_list.append(kpath/qd/f)

print("Long cadence: ")
print(lc_qtr_list)
print()
print("Short cadence: ")
print(sc_qtr_list)

#for qd in os.listdir(kpath):
#   if '_Q' in qd:
#      print(qd)	     
#      for f in os.listdir(qd):
#         print(f)


#           print(f)
#	      if star in f:#
#			 print(f)
      
'''
   for f in $qd/kplr${star}* ; do
	    if [ -f $f ] ; then
		echo $f
	    fi
	done
    f'''

