# conda activate py38
# module load utils/julia
export OMP_NUM_THREADS=2
python3 gen_ham.py && julia runtests.jl && julia h2.jl
