#!/usr/bin/env bash

# Author: Selahattin Baki Damar

set -e

cslc ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=N:6 -o out --memcpy --channels 1
cs_python run.py --name out
