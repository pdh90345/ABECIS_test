# ABECIS_test


## Installation


3. Then, upgrade pip using

```
python3 -m pip install --upgrade pip
```

4. Install torch using

```
pip install torch torchvision torchaudio
```

5. Install detectron2, the instance segmentation framework used by ABECIS

```
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
```
You might also need C++ Build Tools on Windows, get it [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

If there is any issue with pycocotools, get C++ build tools first, then install with 
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```


6. Run the python script named **setup.py** using the following command, to set up the dependencies.

```
python3 ./setup.py
```

**[IMPORTANT]** Don't forget to rerun setup.py to install dependencies afterwards.

7. If everything ran smoothly, run ABECIS by

```
python3 ./abecis.py
```

> Note: When running for the first time, it will automatically download the pre-trained model, and will take some time.
