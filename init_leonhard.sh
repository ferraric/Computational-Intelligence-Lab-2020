module load gcc/6.3.0 python_gpu/3.7.4
module load eth_proxy
source venv/bin/activate
pip3 install -r requirements.txt
pre-commit install
# The pre-commit hook isort looks through all these config files.
# If it does not find them in the project root folder,
# it tries to look for them in the $HOME folder (/cluster/home),
# where it does not have permission and thus fails.
# We create emtpy config files in the project root folder
# in order to prevent that.
touch .editorconfig
touch pyproject.toml
touch .isort.cfg
touch setup.cfg
touch tox.ini
