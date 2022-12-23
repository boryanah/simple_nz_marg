#!/bin/bash

queue=cmb

# Likelihoods

# addqueue -n 2x24 -s -q $queue -m 1  -c "DES 3x2pt LJ CAMB"  -o log/des_3x2pt_limberjackpriors_camb.out       ./launch_cobaya.sh ./input/des_3x2pt_limberjackpriors_camb.yml
addqueue -n 2x24 -s -q $queue -m 1  -c "DES 3x2pt LJ EH"    -o log/des_3x2pt_limberjackpriors_eh.out         ./launch_cobaya.sh ./input/des_3x2pt_limberjackpriors_eh.yml
