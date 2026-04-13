# MultimodalMedicalImaging
Capstone assignment



# Getting Started

## Step 1
https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21
Download the Fold 4 model from this github
## Step 2
Extract the dataset from /common/dataset/synapse/BraTS-2023/BraTS-GLI using 
`mv /common/dataset/synapse/BratS-2023/BraTS-GLI ./ -r`

from there we are going to extract the file.

`UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip filename.zip`

## Step 3
from here edit the foldmaker.sh script with the correct filepaths regarding your system.

run 
`./foldmaker.sh`

## Step 4

Next, change the dropout_experiment.sh variables to the ones to your system.
then run

`./dropout_experiment.sh`