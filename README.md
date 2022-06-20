# ComputerVisionFinalProject

## Create virtual env and connect to VSCode to use it on jupyter notebooks

Also installing requirements:

```ssh
virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name=env
```