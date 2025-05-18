# FaceGen 

**FaceGen** is a deep learning-based application for generating and editing human bust images. It combines the power of **StyleGAN3**, **InterfaceGAN** and **GANSpace**. **StyleCLIP** can be tried by it's standalone file. 

## Features

- Generate realistic human faces using StyleGAN3
- Edit facial attributes using InterfaceGAN and GANSpace
- Perform text-based edits via StyleCLIP (Run gen_faces_styleclip.py for this)

## Example Outputs
![Original Subject](examples/subject_6abb48ef4a/original.png)

![Young Subject](examples/subject_6abb48ef4a/edited_young.png)

![Old Subject](examples/subject_6abb48ef4a/edited_old.png)

![Subject Posing Left](examples/subject_6abb48ef4a/edited_pose_left.png)

![Subject Posing Right](examples/subject_6abb48ef4a/edited_pose_right.png)

![Subject Tilting Up](examples/subject_6abb48ef4a/edited_pose_tilt_down.png)

![Subject Tilting Down](examples/subject_6abb48ef4a/edited_pose_tilt_up.png)

![Original Subject Smiling](examples/subject_6abb48ef4a/edited_smile.png)

![Original Under Different Lighting](examples/subject_6abb48ef4a/edited_light_0.png)

![Original Under Different Lighting](examples/subject_6abb48ef4a/edited_light_1.png)

![Original Under Different Lighting](examples/subject_6abb48ef4a/edited_light_2.png)

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

#To run our version using StyleCLIP simply 
