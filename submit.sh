#!/bin/bash
#SBATCH --account=sdss-np
#SBATCH --partition=sdss-np
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64

#SBATCH --time=24:00:00
#SBATCH --job-name=gaia_apg
#SBATCH --output=%x_%j.out
#SBATCH --err=%x_%j.err

# ------------------------------------------------------------------------------

module load julia

julia random_draw_andrew.jl
