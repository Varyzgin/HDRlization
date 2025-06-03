# Forked from An Gia Vien and Chul Lee

Official PyTorch Code for **"Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging"**

Paper link: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670429.pdf

### Requirements
- 13 GB of disk space

### Set up
- git clone https://github.com/Varyzgin/HDRlization.git
- download and unzip to "Kalantari" folder data for testing: https://drive.google.com/file/d/1bkyNjlMst8rz5xRI43uzkOwhtNiTMWI2/view?usp=sharing
- download and unzip to "WEIGHTS_ECCV2022" folder pretrained weights from: https://drive.google.com/file/d/1bkyNjlMst8rz5xRI43uzkOwhtNiTMWI2/view?usp=sharing
- cd HDRlization
- docker build -t hdr .
- docker run -it hdr
- Test data path (e.g., "Kalantari/")
- Output path (e.g., "test_results/")
- Weight path (e.g., "WEIGHTS_ECCV2022/")

### Citation
Please cite the following paper if you feel this repository useful.
```
    @inproceedings{EDWL,
        author    = {An Gia Vien and Chul Lee}, 
        title     = {Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2022}
    }
```
### License
See [MIT License](https://github.com/viengiaan/EDWL/blob/main/LICENSE)
