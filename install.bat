:: virtual environment installation
python -m venv .env/
.env\Scripts\activate.bat

:: installation dependencies
pip install --upgrade tensorflow
pip install --upgrade tqdm
deactivate

:: get CIFAR10 dataset manually
