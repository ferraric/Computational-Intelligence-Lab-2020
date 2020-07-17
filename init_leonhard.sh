module load gcc/6.3.0 python_gpu/3.7.1 cuda/10.1.243 cudnn/7.6.4 eth_proxy
source venv/bin/activate
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pre-commit install
export PYTHONPATH=$PYTHONPATH:~/Computational-Intelligence-Lab-2020
# to prevent the pre-commit hook "isort" to look through all config files in the $HOME folder 
# (where it doesn't have permission and fails), we create empty config files in the project root folder
touch .editorconfig
touch pyproject.toml
touch .isort.cfg
touch setup.cfg
touch tox.ini
