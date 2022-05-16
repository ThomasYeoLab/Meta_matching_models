# Meta_matching_models v1.0
This folder contains  v1.0 meta matching model trained from the UK Biobank (N = 36,847) and example to run it. This is the model used in Figure 7 of the He et al., 2020.

## Upcoming release
In He et al., 2020, we tested our MM model on datasets with ICA-FIX. The model does not seem to work so well for data that has undergone global signal regression (GSR). We will be releasing a new MM model that also generalizes well on datasets with GSR. Please stay tune.

## Reference
He T, An L, Feng J, Bzdok D, Eickhoff SB, Yeo BTT. **Meta-matching: a simple approach to translate predictive models from big to small data**. BioRxiv, 2020.08.10.245373, under review.

## Data Processing
Meta matching model v1.0 used Schaefer2018 parcellation with 400 parcels and 19 sub-cortical regions. In order to use the pre-trained model, you need generate your own data with same parcellation. You can refer to the [data processing code for HCP dataset in our full release](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/He2022_MM/data_processing#step-51-step-5-for-whole-data-processing-code) where we convert ICA FIX HCP S1200 data into 419 by 419 FC (functional connectivity) matrix with [Schaefer2016_400Parcels_17Networks_colors_19_09_16_subcortical.dlabel.nii](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/data_processing/step5_hcp_data/extra/Schaefer2016_400Parcels_17Networks_colors_19_09_16_subcortical.dlabel.nii), and [data processing code for UK Biobank dataset in our full release](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/He2022_MM/data_processing#step-60-optional) where we convert rsfMRI data [data-field 20227 of UK Biobank](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20227) to MNI152 2mm space then to 419 by 419 FC (functional connectivity) matrix with [Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/data_processing/step6_experiment_2/ukbb_20227_to_fc419/extra/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz). If you need more information about 19 sub-cortical regions, you can take a better look at [code that we created 419 mask for UK Biobank dataset in our full release](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/data_processing/step6_experiment_2/ukbb_20227_to_fc419/CBIG_MM_create_FC419_MNI2mm.m).

The model needs flatten FC for input, here is the simple code to flatten 419 by 419 matrix.
```python
import numpy as np
roi = 419
index = np.tril(np.ones(roi), k=-1) == 1
fc_flat = fc[index]
```

## Usage
Please check meta_matching_v1.0.ipynb for the model usage and example.

## Bugs and Questions
Please contact He Tong at hetong1115@gmail.com, Lijun An at anlijun.cn@gmail.com, Pansheng Chen at chenpansheng@gmail.com and Thomas Yeo at yeoyeo02@gmail.com.
