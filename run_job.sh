bsub -n 1 -R "rusage[mem=8000,ngpus_excl_p=1]" -W 10:00 python main.py
