module load gcc/6.3.0 python_gpu/3.7.4
module load eth_proxy
source venv/bin/activate
pip3 install -r requirements.txt
pre-commit install
# to prevent the pre-commit hook "isort" to look through all config files in the $HOME folder 
# (where it doesn't have permission and fails), we create empty config files in the project root folder
touch .editorconfig
touch pyproject.toml
touch setup.cfg
touch tox.ini
