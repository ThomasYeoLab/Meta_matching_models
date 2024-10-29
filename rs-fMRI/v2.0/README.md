# Meta_matching_models v2.0 (Multilayer Meta-matching)
This folder contains v2.0 (multilayer meta-matching) models trained from 5 source datasets (UK Biobank, ABCD, GSP, HBN, eNKI-RS) and example to run it. This repo contains pre-trained Meta-matching models. If you want to train your own multilayer meta-matching models from scratch or view more details, please visit our [CBIG repo](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/Chen2024_MMM).


## Reference
+ Chen, P., An, L., Wulan, N., Zhang, C., Zhang, S., Ooi, L. Q. R., ... & Yeo, B. T. (2024). [**Multilayer meta-matching: translating phenotypic prediction models from multiple datasets to small data**](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00233/123369/Multilayer-meta-matching-Translating-phenotypic). Imaging Neuroscience, 2, 1-22.


## Download models from the GiHub relase page
Since GitHub has restrictions on uploading large files, we put all meta-matching v2.0 models in [our GitHub release page](https://github.com/ThomasYeoLab/Meta_matching_models/releases/tag/v2.0-rsfMRI). After cloning our source code, you need to download v2.0 models (meta-matching.models.v2_0.rar), unzip it, and put model files under "Meta_matching_models/rs-fMRI/v2.0/models/"

## Data processing
Meta-matching model v2.0 used [Schaefer2018 parcellation with 400 cortical regions](https://doi.org/10.1093/cercor/bhx179) ([Yeo17 network](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/Yeo2011_17networks_N1000.split_components.glossary.csv) default ordering) and [19 sub-cortical regions](https://doi.org/10.1016/s0896-6273(02)00569-x) based on FreeSurfer (the order is: Left-Cerebellum-Cortex, Left-Thalamus-Proper, Left-Caudate, Left-Putamen, Left-Pallidum, Brain-Stem, Left-Hippocampus, Left-Amygdala, Left-Accumbens-area, Left-VentralDC, Right-Cerebellum-Cortex, Right-Thalamus-Proper, Right-Caudate, Right-Putamen, Right-Pallidum, Right-Hippocampus, Right-Amygdala, Right-Accumbens-area, Right-VentralDC).

In order to use the pre-trained model, **you need generate your own data with same parcellation and same ROI order.** You can refer to Section 2.1 of [Chen et al. 2024](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00233/123369/Multilayer-meta-matching-Translating-phenotypic) for detailed information of data processing of all datasets. But basically, for HCP-YA and HCP-Aging, we used [Schaefer 400 parcellation on fsLR_32k space](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii); For UK Biobank and eNKI, we used [Schaefer 400 parcellation on MNI space](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/data_processing/step6_experiment_2/ukbb_20227_to_fc419/extra/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz); And for ABCD, GSP, HBN, we used Schaefer 400 parcellation on fsaverage6 space ([lh.Schaefer2018_400Parcels_17Networks_order.annot](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage6/label/lh.Schaefer2018_400Parcels_17Networks_order.annot) and [rh.Schaefer2018_400Parcels_17Networks_order.annot](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage6/label/rh.Schaefer2018_400Parcels_17Networks_order.annot))

The model needs flatten FC for input, here is the simple code to flatten 419 by 419 matrix.
```python
import numpy as np
roi = 419
index = np.tril(np.ones(roi), k=-1) == 1
fc_flat = fc[index]
```
## Source phenotypes in each source dataset
The description of different source phenotypes can be found in Supplementary Tables S2 - S6 of [Chen et al. 2024](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00233/123369/Multilayer-meta-matching-Translating-phenotypic) 

The phenotype sequence of the UK Biobank DNN are the same as v1.1 and v1.0 (see Table S2 of [Chen et al. 2024](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00233/123369/Multilayer-meta-matching-Translating-phenotypic) ):

|        Outputs of DNN        | <span style="font-weight:normal"> #1 | <span style="font-weight:normal"> #2 | <span style="font-weight:normal"> #3 | <span style="font-weight:normal"> #4 | <span style="font-weight:normal"> #5  | <span style="font-weight:normal"> #6 | <span style="font-weight:normal"> #7  | <span style="font-weight:normal"> #8 | <span style="font-weight:normal"> #9  | <span style="font-weight:normal"> #10 | <span style="font-weight:normal"> #11  | <span style="font-weight:normal"> #12 | <span style="font-weight:normal"> #13  | <span style="font-weight:normal"> #14 | <span style="font-weight:normal"> #15  | <span style="font-weight:normal"> #16 | <span style="font-weight:normal">  #17 | <span style="font-weight:normal">#18 | <span style="font-weight:normal">#19 | <span style="font-weight:normal">#20 | <span style="font-weight:normal">#21 | <span style="font-weight:normal">#22 | <span style="font-weight:normal">#23 | <span style="font-weight:normal">#24 | <span style="font-weight:normal">#25 | <span style="font-weight:normal">#26 | <span style="font-weight:normal">#27 | <span style="font-weight:normal">#28 | <span style="font-weight:normal">#29 | <span style="font-weight:normal">#30 | <span style="font-weight:normal">#31 | <span style="font-weight:normal">#32 | <span style="font-weight:normal">#33 | <span style="font-weight:normal">#34 | <span style="font-weight:normal">#35 |<span style="font-weight:normal"> #36 | <span style="font-weight:normal">#37 |<span style="font-weight:normal"> #38 | <span style="font-weight:normal">#39 |<span style="font-weight:normal"> #40 | <span style="font-weight:normal">#41 | <span style="font-weight:normal">#42 | <span style="font-weight:normal">#43 |<span style="font-weight:normal"> #44 |<span style="font-weight:normal"> #45 |<span style="font-weight:normal"> #46 | <span style="font-weight:normal">#47 | <span style="font-weight:normal">#48 | <span style="font-weight:normal">#49 | <span style="font-weight:normal">#50 |<span style="font-weight:normal"> #51 |<span style="font-weight:normal"> #52 | <span style="font-weight:normal">#53 | <span style="font-weight:normal">#54 | #55 | <span style="font-weight:normal">#56 | <span style="font-weight:normal">#57 | <span style="font-weight:normal">#58 | <span style="font-weight:normal">#59 | <span style="font-weight:normal">#60 | <span style="font-weight:normal">#61 | <span style="font-weight:normal">#62 | <span style="font-weight:normal">#63 | <span style="font-weight:normal">#64 |<span style="font-weight:normal"> #65 | <span style="font-weight:normal">#66 |   <span style="font-weight:normal">   #67 |
|:-----------------------:|:------------------------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **Corresponding phenotypes** |             Alcohol 1             |              Alcohol 2               |              Alcohol 3               |              Time walk               |              Time drive               |               Time TV                |                 Sleep                 |               Age edu                |                 Work                  |                Travel                 |               #household               |                 Neuro                 |                Hearing                 |              Fluid Int.               |                Matching                |                  Sex                  |               Matching-o               | Age| Trail-o C1| Trail-o C3| Digit-o C1| Digit-o C6| Sex G C1| Sex G C2| Genetic C1| Cancer C1| Urine C1| Blood C2| Blood C3| Blood C4| Blood C5| Deprive C1| Dur C1| Dur C2| Dur C4| Trail C1| Tower C1| Digit 1| Match| ProMem C1| #Mem C1| Matrix C1| Matrix C2| Matrix C3| Illness C1| Illness C4| Loc C1| Breath C1| Grip C1| ECG C1| ECG C2| ECG C3| ECG C6| Carotid C1| Carotid C5| Bone C1| Bone C3| Body C1| Body C2| Body C3| BP eye C2| BP eye C3| BP eye C4| BP eye C5| BP eye C6| Family C1| Smoke C1 |


The phenotype name of Ridge Regression models are specified in corresponding model files (end with '.sav').
```python
# Load RR model, here we use models from ABCD for example
dataset = 'ABCD' # source dataset
# models_1layer take FC as input and predicts 36 ABCD phenotypes prediction
models_1layer = pickle.load(open(os.path.join(model_v20_path, dataset + '_rr_models_base.sav'), 'rb'))
# models_2layer takes phenotypic predictions from UK Biobank as input and predicts 36 ABCD phenotypes
models_2layer = pickle.load(open(os.path.join(model_v20_path, dataset + '_rr_models_multilayer.sav'), 'rb'))
# each model file contain multiple RR model as a form of dictionary
# keys are phenotypes names, values are ridge regression model parameters
print(models_1layer.keys())
```

## Usage
Please check meta_matching_v2.0.ipynb for the model usage and example.

## Bugs and Questions
Please contact Pansheng Chen at chenpansheng@gmail.com, Lijun An at anlijun.cn@gmail.com, Chen Zhang at chenzhangsutd@gmail.com and Thomas Yeo at yeoyeo02@gmail.com.

