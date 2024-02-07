# FF-guided-NeuralCC

## 1. Download Datasets
First, download the datasets to be used in the experiment. Here are the download instructions of each dataset.

### CrowdFlow
Download the CrowdFlow dataset from the link below. The password is the name of the dataset. More details can be found in the official CrowdFlow repository [tsenst/CrowdFlow](https://github.com/tsenst/CrowdFlow).
- [https://hidrive.ionos.com/lnk/LUiCHfYG](https://hidrive.ionos.com/lnk/LUiCHfYG)

### FDST
Download the FDST dataset from the link below. More details can be found in the official FDST repo [sweetyy83/Lstn_fdst_dataset](https://github.com/sweetyy83/Lstn_fdst_dataset)
- [https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w](https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w)

### CityStreet
Download the CityStreet dataset from the link below. More details can be found in the official HP [http://visal.cs.cityu.edu.hk/downloads/citystreetdata/](http://visal.cs.cityu.edu.hk/downloads/citystreetdata/).
- [https://drive.google.com/drive/folders/11hK1REG3P35S9ANXk1YB7C1-_SS_LQGJ](https://drive.google.com/drive/folders/11hK1REG3P35S9ANXk1YB7C1-_SS_LQGJ)

## 2. Make labels and json files
Next, create labels for the density map. There are two ways to generate labels, each treated as a separate dataset. Dataset A contains labels that map the binary value indicating the presence or absence of a person to a single cell, while dataset B contains labels that map the count of people.

The command below shows an example of creating a label for Crowdflow dataset A.
```sh
python create_labels.py CrowdFlow --path /path/to/CrowdFlow --mode once
```

## 3. Train model

```sh
python train.py A_train_add.csv A_val_add.csv --dataset CrowdFlow --exp /groups1/gca50095/aca10350zi/habara_exp/cross_val_A_add/ --myloss 0
```

## 4. Tuning hyperparameters of floor-field

```sh
python FF_search_param.py data/Scene_IM04.csv data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${c}/${s}/model_best.pth.tar --DynamicFF 1 --StaticFF 1
```

## 5. Test

```sh
python FF_test.py data/Scene_IM04.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${c}/${s}/model_best.pth.tar --DynamicFF 1 --StaticFF 1
```
