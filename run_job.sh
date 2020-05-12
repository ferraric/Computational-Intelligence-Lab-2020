bsub -n 1 -R "rusage[mem=32000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" -W 120:00 python main.py
