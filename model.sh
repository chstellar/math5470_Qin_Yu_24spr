#!/bin/bash
#
#SBATCH --job-name=5470
#SBATCH -p horence,owners
#SBATCH --time=0-04:00:00 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chesteryu@stanford.edu

ml python/3.9.0
ml glib/2.52.3

python model.py