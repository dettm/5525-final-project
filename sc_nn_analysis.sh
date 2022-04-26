#!/bin/bash
#SBATCH --job-name=nn_analysis
#SBATCH --ntasks=28
#SBATCH --time=12:00:00

python3 -m venv final_env
source final_env/bin/activate
pip install torch
pip install pandas
pip install scikit-learn
pip install nltk
pip install numpy
pip install matplotlib
python3 spam_collection_supercomputer.py