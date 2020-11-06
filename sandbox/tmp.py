import os
import sys

if len(sys.argv) != 2:
   print(f'usage {argv[0]} star_id')
   quit()


star=sys.argv[1]

kpath = '/home/echo/hdd6tb/02_kepler_time_series_scripts/'

qdlis = []
for qd in os.listdir(kpath):
   if '_Q' in qd:
      print(qd)
      for f in os.listdir(kpath+qd):
         print(f)
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

