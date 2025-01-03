# UW-Madison-StomachSegmentation-MRI
This repository hosts the code for a medical image processing project aimed at segmenting the stomach in MRI scans using a binary classification approach. The project utilizes deep learning, specifically the U-Net architecture, to accurately delineate the stomach area in medical images of cancer patients undergoing radiation therapy.

## Project Overview
Radiation therapy for gastro-intestinal cancers requires precise targeting to maximize the treatment efficacy and minimize damage to healthy organs. This project enhances the precision of such treatments by automating the segmentation of the stomach in MRI scans, significantly accelerating the preparation process and reducing the treatment duration for patients.

The model is trained on a dataset of MRI medical images where each scan is labeled with regions of interest, including the stomach. By focusing exclusively on the stomach, the project simplifies the complex task to a binary classification problem, where the presence of stomach tissue is segmented from the rest of the abdominal contents.

## Dataset
The dataset utilized for training the model consists of annotated MRI scans and is accessible via the following link:
[Stomach MRI Dataset](https://drive.google.com/file/d/1JNmn7baTgrEBpHu83y5ckYZ6_rUmBgJp/view)

![case2_day3_slice_0045_image](https://github.com/user-attachments/assets/48b954cc-7545-4fe4-9dd2-a6969c979395)
![case2_day3_slice_0045_mask](https://github.com/user-attachments/assets/b8b8db6f-3611-403a-8a99-51f89437468f)

&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;MR Image&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;Mask

## Model and Training
The segmentation model is based on the U-Net architecture, a convolutional network originally designed for biomedical image segmentation. This architecture is well-suited for tasks like ours because it efficiently handles the variability in medical images and provides precise localization of segmented areas. Transfer learning techniques are also employed, utilizing pre-trained models to enhance our model's performance.

## Usage
Instructions for setting up the project environment, installing dependencies, and executing the model will be provided, ensuring that researchers and practitioners can easily replicate and extend the findings of this project.

## Contributions
Contributions to enhance this project are encouraged. You can help by:
- Improving the model's segmentation accuracy and operational efficiency
- Expanding the dataset with more diverse annotated MRI scans
- Optimizing the code for broader application in medical image analysis

