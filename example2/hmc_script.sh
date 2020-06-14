#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J hmc_example2
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 10GB
### -- set walltime limit: hh:mm --
#BSUB -W 100:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u teekad@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

# here follow the commands you want to execute
module load FEniCS/2019.1.0-with-petsc-3.10.5-and-numpy-1.16.5
#module load matplotlib/2.0.2-python-3.6.2

#one long line:
export PYTHONPATH=$HOME/local/multiphenics/lib/python3.6/site-packages/:$PYTHONPATH

#another long line:
export PYBIND11_DIR=/appl/FEniCS/2019.1.0.with.petsc.3.10.5.and.numpy.1.16.5/$CPUTYPEV

python3 main_problem_hmc.py
