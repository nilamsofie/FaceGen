# FaceGen 

**FaceGen** is a deep learning-based application for generating and editing human bust images. It combines the power of **StyleGAN3**, **InterfaceGAN** and **GANSpace**. **StyleCLIP** can be tried by it's standalone file. 

## Features

- Generate realistic human faces using StyleGAN3
- Edit facial attributes using InterfaceGAN and GANSpace
- Perform text-based edits via StyleCLIP (Run gen_faces_styleclip.py for this)

## Technologies Used

- [StyleGAN3](https://github.com/NVlabs/stylegan3)
- [InterfaceGAN](https://github.com/ShenYujun/InterfaceGAN)
- [GANSpace](https://github.com/harskish/ganspace)
- [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
- Python 

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/nilamsofie/FaceGen.git
cd FaceGen

# Create the required conda environment
conda env create -f stylegan3/environment.yml
conda activate stylegan3

# Run the following command
python gen_faces.py --n_subjects 30 # Replace the number according to the number of subjects you want
