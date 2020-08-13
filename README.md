# An implementation of GLapGAN: "Performing Group Difference Testing on Graph Structured Data from GANs: Analysis and Applications in Neuroimaging"
This is an PyTorch implementation of GLapGAN as described in the paper [Performing Group Difference Testing on Graph Structured Data from GANs: Analysis and Applications in Neuroimaging]

# Training Recipe 
1. number of epochs: 500
2. initial learning rates (1e-5, 5*1e-5) for generator and discriminator

# Datasets
Processed ADNI data can not be released without special permission in public. Please email us for the whole processed adni data. 
Here, we provide two samples (one health and one disease) for understanding the input data. The two-sample ADNI data can be downloaded at [here](https://drive.google.com/drive/folders/1aa5PCcO6Q5W91BERt6yiSImG-dkhKHmY?usp=sharing). Then put it under
src folder. 

# Trained models
Trained models can be downloaded at [here](https://drive.google.com/drive/folders/1aa5PCcO6Q5W91BERt6yiSImG-dkhKHmY?usp=sharing).

# Usage
cd src && python demo_lwgan.py --flag_ttest --num_ads 183 --num_cns 293 --lr_g 0.00001 --lr_d 0.00005 --gpu 0 --flag_reg --result_path ./results/lwgan_adni_293_reg

# Reference

