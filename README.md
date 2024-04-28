# Environment
***
1. python = 3.9.10
2. pytorch = 1.12.1+cu113
3. tqdm = 4.64.1
4. opencv-python = 4.5.3.56
5. Linux(ubuntu)

# Training
***
1. The pre-trained models are in "checkpoints/GMM-GAN";
2. Download the test samples from <u>[GoogleDrive](https://drive.google.com/drive/folders/1YZfmo-0d-0gl97soTPnnnXjUqPJvqUzd?usp=sharing)</u> and place them in the appropriate dir ("data/IXI/val or data/BraTS2015/val or data/OASIS3/val");
3. Modify the GMM-GAN.yaml in "Yaml/GMM-GAN.yaml".When testing with the IXI, OASIS3 dataset```input_nc=3``` ; When testing with the BraTS2015 dataset```input_nc=1```;
4. ```python
   python test.py
