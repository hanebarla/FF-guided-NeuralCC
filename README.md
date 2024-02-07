# FF-guided-NeuralCC

## 1. Download Datasets
First, download the datasets to be used in the experiment. Here are the download instructions of each dataset.

### CrowdFlow
Download the CrowdFlow dataset from the link below. The password is the name of the dataset. More details can be found in the official CrowdFlow repository [tsenst/CrowdFlow](https://github.com/tsenst/CrowdFlow).
- [https://hidrive.ionos.com/lnk/LUiCHfYG](https://hidrive.ionos.com/lnk/LUiCHfYG)

### FDST
Download the FDST dataset from the link below. More details can be found in the official FDST repo [sweetyy83/Lstn_fdst_dataset](https://github.com/sweetyy83/Lstn_fdst_dataset).
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
After creating labels, the model is trained. The --penalty argument allows us to scale our proposed penalty term during training.

The command below shows an example of training CAN mode with CrowdFlow datast A, using MSE loss and our penalty term.
```sh
python train.py A_train_once.csv A_val_once.csv --dataset CrowdFlow --exp /path/to/save_dir --penalty 0.1
```

## 4. Tuning hyperparameters of floor-field
If the floor-field model is to be used to correct the model's output, the hyperparameters need to be tuned before its evaluation.

The command below shows an example of tuning hyperparameters of static and dynamic floor field's hyperparameters with CrowdFlow datast A.
```sh
python ff_search_param.py A_test_once.csv --dataset CrowdFlow --saved_dir /path/to/saved_dir --DynamicFF 1 --StaticFF 1
```

## 5. Test

The command below shows an example of testing of both floor fields with CrowdFlow datast A.
```sh
python ff_test.py A_test_once.csv --dataset CrowdFlow --saved_dir /path/to/saved_dir --DynamicFF 1 --StaticFF 1
```
