#!/bin/bash
export OMP_NUM_THREADS=8

param=$1

/usr/local/shared/slurm/bin/srun -N 2 -n 6 --ntasks-per-node 3 -m cyclic --mpi=pmi2 /usr/bin/python3.6 -m cobaya run $param
# /usr/local/shared/slurm/bin/srun -n 6 --mem-per-cpu 7000 -m cyclic --mpi=pmi2 ./run_cl_cross_corr_v3_blcdm $model 10000000 $new
