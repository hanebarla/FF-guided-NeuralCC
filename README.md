# Floor-Field-Guided Neural Model for Crowd Counting

We proposed a method that combines neural networks with crowd dynamics. Specifically, we introduced a loss function to represent prior knowledge of this dynamics and propose static/dynamic floor field models. In the main manuscript, we discussed the effectiveness of these methods through numerical experiments. To ensure reproducibility and transparency of our experiments, we made the scripts used for evaluation available in this repository.

<!-- 我々はニューラルネットワークと群衆ダイナミクスを組み合わせた手法を提案した．具体的には，このダイナミクスの事前知識を表す損失関数と静的/動的フロアフィールドモデルを提案し，本文ではその効果について数値実験の結果を通じて詳しく論じた．その実験の再現性・透明性を確保するために，評価に用いたスクリプトを本レポジトリに公開する． -->
1. [Prepare an environment](#1-prepare-an-environment)
1. [Make labels and json files](#2-make-labels-and-json-files)
1. [Train model](#3-train-model)
1. [Tune hyperperameters of floor field models](#4-tune-hyperparameters-of-floor-field)
1. [Test](#5-test)

## 1. Prepare an environment
First, download the datasets to be used in the experiment. Here are the download instructions of each dataset.

#### CrowdFlow
Download the CrowdFlow dataset from the link below. The password is the name of the dataset. More details can be found in the official CrowdFlow repository [tsenst/CrowdFlow](https://github.com/tsenst/CrowdFlow).
- [https://hidrive.ionos.com/lnk/LUiCHfYG](https://hidrive.ionos.com/lnk/LUiCHfYG)

#### FDST
Download the FDST dataset from the link below. More details can be found in the official FDST repo [sweetyy83/Lstn_fdst_dataset](https://github.com/sweetyy83/Lstn_fdst_dataset).
- [https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w](https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w)

#### CityStreet
Download the CityStreet dataset from the link below. More details can be found in the official HP [http://visal.cs.cityu.edu.hk/downloads/citystreetdata/](http://visal.cs.cityu.edu.hk/downloads/citystreetdata/).
- [https://drive.google.com/drive/folders/11hK1REG3P35S9ANXk1YB7C1-_SS_LQGJ](https://drive.google.com/drive/folders/11hK1REG3P35S9ANXk1YB7C1-_SS_LQGJ)


Then, set up the program execution environment. The language used is Python 3.10, and if you are using GPU as a hardware accelerator, please make sure to use CUDA 11.7. You can install the required external libraries using the following command:
```sh
pip install -r requirements.txt
```
This command will allow you to install all the necessary external libraries. Make sure to execute this command to ensure that all dependencies are properly installed.

## 2. Make labels and json files
Next, create labels for the density map using ```create_labels.py```. There are two ways to generate labels, each treated as a separate dataset. Dataset A contains labels that map the binary value indicating the presence or absence of a person to a single cell, while dataset B contains labels that map the count of people. The ```create_labels.py``` is used to create the label. The first argument is a non-optional argument that indicates the dataset for which the label is to be created. The ```--path``` argument indicates the directory where the dataset is stored, and ```--mode``` can be used to specify whether to create labels for dataset A or B. If ```once``` is specified, labels can be created for DataSet A. If ```add``` is specified, labels can be created for DataSet B.

The command below shows an example of creating a label for Crowdflow dataset A.
```sh
python create_labels.py CrowdFlow --path /path/to/CrowdFlow --mode once
```
If you want to create labels of the FDST or CityStreet datasets, please replace "CrowdFlow" in the first argument with "fdst" or "citystreet", respectively. 

<!-- このプログラムの実行により，さらに学習・検証・テストデータに分割し，そのファイルをまとめたjsonファイルを得ることができる．jsonファイル名は```[dataset name]_[label type]_[use case].json```であり，```[dataset name]```はcrowdflowやfdstといったデータセット名が記載され，```[label type]```は```once```か```add```のどちらか，```[use case]```は学習用，検証用，テスト用のいずれかが表記される． -->
By running this program, you can further split the data into training, validation, and testing sets, and obtain a JSON file containing the file paths. The JSON file name follows the format ```[dataset name]_[label type]_[use case].json```, where ```[dataset name]``` indicates the dataset name such as "crowdflow" or "fdst", ```[label type]``` is either "once" or "add", and ```[use case]``` denotes whether it is for training ("train"), validation ("val"), or testing ("test").

## 3. Train model
<!-- After creating the labels, the model is trained using the ```train.py```.　最初の2つの引数必須引数で，学習・検証データのパスがまとめられたjsonファイルを指定できる．また```--exp```引数は実験結果を保存するディレクトリをを指定できる． The ```--penalty``` argument allows scaling the proposed penalty term during training. The default value of the argument is ```0```. -->
After creating the labels, the model is trained using `train.py`. The first two arguments are mandatory, where you specify the JSON file containing the paths to the training and validation data. Additionally, the `--exp` argument allows you to specify the directory to save the experimental results. The `--penalty` argument allows scaling the proposed penalty term during training, with the default value being "0".

The command below shows an example of training a Extended CAN model with CrowdFlow datast A, using MSE loss and our penalty term.
```sh
python train.py crowdflow_once_train.json crowdflow_once_val.json --exp /path/to/save_dir --penalty 0.1
```

## 4. Tune hyperparameters of floor field models
If the floor-field model is used, the hyperparameters need to be tuned before its evaluation. The ```ff_search_param.py``` file can tune the hyperparameters. The ```--dynamicff 1``` allows us to use dynamic floor field, and the ```--staticff 1``` allows us to use static floor field. The default value of the both argument is ```0```. 

The command below shows an example of tuning hyperparameters of static and dynamic floor field's hyperparameters with CrowdFlow datast A.
```sh
python ff_search_param.py crowdflow_once_train.json --saved_dir /path/to/saved_dir --dynamicFF 1 --staticFF 1
```

## 5. Test
<!-- 最後に提案手法の有効性を確認するため，```ff_test.py```を用いて評価する．このプログラムの引数は```ff_search_param.py```と同様である． -->
Finally, to verify the effectiveness of the proposed method, you can use ```ff_test.py``` for evaluation. The arguments for this program are the same as those for ```ff_search_param.py```.

The command below shows an example of testing of both floor fields with CrowdFlow datast A.
```sh
python ff_test.py crowdflow_once_test.json --saved_dir /path/to/saved_dir --dynamicFF 1 --staticFF 1
```
