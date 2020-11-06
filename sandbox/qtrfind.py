import os
import sys

if len(sys.argv) != 2:
   print(f'usage {argv[0]} star_id')
   quit()


star=sys.argv[1]

kpath = '/home/echo/hdd6tb/02_kepler_time_series_scripts/'
qtrlis  = []
for qd in os.listdir(kpath):
   if '_Q' in qd:
      for f in os.listdir(kpath+qd):
         if star in f:#
            qtrlis.append(kpath+qd+'/'+f)
#            print(kpath+qd+'/'+f)

for q in qtrlis:
   print(q)

