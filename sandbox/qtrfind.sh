#!/bin/sh

if [ $# -ne 1 ] ; then
    echo usage $0 star_id
    exit
fi

star=$1

for qd in $HOME/hdd6tb/02_kepler_time_series_scripts/*Kepler_Q* ; do
    if [ -d $qd ] ; then
	for f in $qd/kplr${star}* ; do
	    if [ -f $f ] ; then
		echo $f
	    fi
	done
    fi
done

