# FaceGen 

**FaceGen** is a deep learning-based application for generating and editing human bust images. It combines the power of **StyleGAN3**, **InterfaceGAN** and **GANSpace**. **StyleCLIP** can be tried by it's standalone file. 

## Features

- Generate realistic human faces using StyleGAN3
- Edit facial attributes using InterfaceGAN and GANSpace
- Perform text-based edits via StyleCLIP (Run gen_faces_styleclip.py for this)

## Example Outputs Using gen_faces.py

<figure>
  <img src="examples/subject_6abb48ef4a/original.png" alt="Original Subject" width="300">
  <figcaption>Original Subject</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_smile.png" alt="Original Subject Smiling" width="300">
  <figcaption>Original Subject Smiling</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_young.png" alt="Young Subject" width="300">
  <figcaption>Young Subject</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_old.png" alt="Old Subject" width="300">
  <figcaption>Old Subject</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_pose_left.png" alt="Subject Posing Left" width="300">
  <figcaption>Subject Posing Left</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_pose_right.png" alt="Subject Posing Right" width="300">
  <figcaption>Subject Posing Right</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_pose_tilt_up.png" alt="Subject Tilting Up" width="300">
  <figcaption>Subject Tilting Up</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_pose_tilt_down.png" alt="Subject Tilting Down" width="300">
  <figcaption>Subject Tilting Down</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_light_0.png" alt="Original Under Different Lighting" width="300">
  <figcaption>Original Under Different Lighting</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_light_1.png" alt="Original Under Different Lighting" width="300">
  <figcaption>Original Under Different Lighting</figcaption>
</figure>

<figure>
  <img src="examples/subject_6abb48ef4a/edited_light_2.png" alt="Original Under Different Lighting" width="300">
  <figcaption>Original Under Different Lighting</figcaption>
</figure>

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

# For StyleCLIP you will also need the following
pip install git+https://github.com/openai/CLIP.git

# Run the following command
python gen_faces.py --n_subjects 30 # Replace the number according to the number of subjects you want

#To run our version using StyleCLIP simply run gen_faces_styleclip.py instead of gen_faces.py
