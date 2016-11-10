




######ECONWEEK7.SH######




#!/bin/bash
 
#SBATCH --time=00:15:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -C 'intel'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
 
# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch
 
# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
 
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
 
module load python/2.7.9
 
outfile="file.o"
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
mpiexec -np 5 python Problem1.py > "$outfile"
mpiexec -np 5 python Problem2.py > "$outfile"
mpiexec -np 5 python Problem3.py 100> "$outfile"
mpiexec -np 5 python Problem4.py 0 1 100 > "$outfile"
mpiexec -np 5 python Problem5.py 100 > "$outfile"
mpiexec -np 5 python Problem6.py 100 > "$outfile"
mpiexec -np 5 python Problem7.py 100 > "$outfile"
mpiexec -np 5 python Problem8.py 10 2 > "$outfile"
mpiexec -np 5 python Problem9.py 100 > "$outfile"
mpiexec -np 5 python Problem10.py 0 1 100 > "$outfile"
mpiexec -np 5 python Problem11.py 6 > "$outfile"
mpiexec -np 5 python Problem12.py 5 5 > "$outfile"
mpiexec -np 5 python Problem13.py 2 6 > "$outfile"
mpiexec -np 5 python Problemextra10.py 100 > "$outfile"
mpiexec -np 5 python Problemextra15.py 100 > "$outfile"
 
exit 0
