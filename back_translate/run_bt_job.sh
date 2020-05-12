module load python_gpu/3.6.4
bsub -n 4 -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" -W 120:00 "source venv/bin/activate; bash run.sh"