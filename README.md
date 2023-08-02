# Multi-Modal Face Anti-Spoofing Ensemble

This repository contains an ensemble of different techniques for face anti-spoofing, combining approaches such as binary pixel-wise segmentation, patch analysis, depth-based analysis, and deepfake detection. It aims to provide a comprehensive solution to detecting fake or spoofed faces in images.

## Folders

- **Binary_Pixel**: Contains the implementation of a binary pixel-wise segmentation approach for anti-spoofing.

- **Patch_Depth**: Implements a patch and depth-based approach for anti-spoofing.

- **MesoNet**: Implements a deepfake detection technique using [MesoNet](https://github.com/DariusAf/MesoNet).

- **Ensemble(Final)**: Combines and integrates the above approaches into an ensemble for improved accuracy.

- **MTCNN_Face_Extraction**: Implements face extraction using the [MTCNN](https://github.com/ipazc/mtcnn) model.

- **PRNet**: Implements depth-based face construction using [PRNet](https://github.com/yfeng95/PRNet).

- **SimSwap**: Not an added folder but used to generate deepfake data on our dataset. Uses [SimSwap](https://github.com/neuralchen/SimSwap) repo.
  
## Requirements
Try to run the code in a conda environment
```shell
conda create --name fas python=3.8
conda activate fas
```

In case of a dlib error 
```shell
pip install cmake
conda install -c conda-forge dlib
```

To run the code in this repository, you need to install the required dependencies listed in the `requirements.txt` file.

```shell
pip install -r requirements.txt
```

## Requirements differences in Mac
- Upgrade macOS to version 12+
- Remove torch from requirements.txt
- Install [miniconda](https://www.youtube.com/watch?v=BEUU-icPg78)
- Run the following before following the other conda instructions
```shell
conda config --set auto_activate_base false
```
- Brew install the rust compiler
```shell
brew install cmake protobuf rust
```
- Install [Tensorflow](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) and Pytorch using the follows
```shell
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
conda install pytorch torchvision torchaudio -c pytorch
```
- Other installations which might be required for installation: hdf5, h5py, chardet, scipy

## Running the code
- Create the environment from the above instructions
- Make [Ensemble(Final)](./Ensemble(Final)) as current directory.
- Create a folder <em>test_img_folder</em> and add images to it
- Run the command
```shell
python ensemble.py
```
