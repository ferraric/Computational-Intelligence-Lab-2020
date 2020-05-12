module load python_gpu/3.6.4
virtualenv venv
source venv/bin/activate
python -m pip install --user -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
deactivate
bash download.sh