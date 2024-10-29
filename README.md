# ABECIS_test

Installation
Before starting, have the sudo access using

sudo -s
For everything below, if 'python3' does not work, replace 'python3' with 'python'

Download Git if you do not have it. And clone this repository using
git clone 'https://github.com/Pi-31415/ABECIS'
and change working directory with

cd ABECIS
Download Python 3 at Python Website, and install.

Then, upgrade pip using

python3 -m pip install --upgrade pip
Install torch using
pip3 install torch torchvision torchaudio
Install detectron2, the instance segmentation framework used by ABECIS
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
You might also need C++ Build Tools on Windows, get it here

If there is any issue with pycocotools, get C++ build tools first, then install with

pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
Run the python script named setup.py using the following command, to set up the dependencies.
python3 ./setup.py
[IMPORTANT] Don't forget to rerun setup.py to install dependencies afterwards.

If everything ran smoothly, run ABECIS by
python3 ./abecis.py
Note: When running for the first time, it will automatically download the pre-trained model, and will take some time.
