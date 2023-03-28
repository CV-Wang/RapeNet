# RapeNet
Crop counting research

Official Pytorch implementation of the paper Automatic rape flower cluster counting method based on low-cost labelling and UAV-RGB images
 (Plant Methods).
 
 We explored a rape flower cluster counting network series: RapeNet and RapeNet+. A rectangular box labeling-based rape flower clusters dataset(RFRB) and a centroid labeling-based rape flower clusters dataset(RFCP) were used for network model training.To verify the performance of RapeNet series, the paper compares the counting result with the real values of manual annotation.The average accuracy(Acc), relative root mean square error(rrMSE) and R2 of the metrics are up to 0.9062, 12.03 and 0.9635 on the dataset RFRB, and 0.9538, 5.61 and 0.9826 on the dataset RFCP, respectively. The resolution has little influence for the proposed model.In addition, the visualization results have some interpretability.Extensive experimental results demonstrate that the RapeNet series outperforms other state-of-the-art counting approaches.The proposed method provides an important technical support for the crop counting statistics of rape flower clusters in field.


The prediction results of RapeNet network are as follows(RFRB/YT4.png, RFCP/202103136307.png)：
![](https://github.com/CV-Wang/RapeNet/blob/main/pred/RapeNet/YT4.png) ![](https://github.com/CV-Wang/RapeNet/blob/main/pred/RapeNet/202103136307.png)

The prediction results of RapeNet+ network are as follows(RFRB/YT4.png, RFCP/202103136307.png)：
![](https://github.com/CV-Wang/RapeNet/blob/main/pred/RapeNet%2B/YT4.png) ![](https://github.com/CV-Wang/RapeNet/blob/main/pred/RapeNet%2B/202103136307.png)


## Prerequisites

Python 3.x

Pytorch >= 1.2

For other libraries, check requirements.txt.

## Getting Started
1. Dataset download

+ RFRB can be downloaded [Google Drive](https://drive.google.com/drive/folders/1HukeRMCmzVWI5uuoymTJ06_jGCXGNxp9?usp=share_link)

+ RFCP can be downloaded [Google Drive](https://drive.google.com/drive/folders/165Ds7MKyaETOyDw1ilOCwYeGhGQgHTuc?usp=share_link)

2. Data preprocess

Due to large sizes of images in RFRB and RFCP datasets, we preprocess these two datasets.

```
python preprocess_RFRB.py
python preprocess_RFCP.py
```

3. Training

```
python train.py --data-dir <path to dataset> --device <gpu device id>
```

4. Test

```
python test_RFRB.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset>
python test_RFCP.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset>
```

## Pretrained models

Pretrained models on RFRB, RFCP can be found [Google Drive](https://drive.google.com/drive/folders/1wz9c4wUQB7-wtd4W3mNPsC6dAyHBcFlc?usp=share_link). You could download them and put them in ckpt folder.
