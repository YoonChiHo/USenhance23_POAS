# USenhance23_POAS
## USenhance23 Challenge: https://ultrasoundenhance2023.grand-challenge.org/
## Dataset Preparation
- [Train/Test Dataset](https://postechackr-my.sharepoint.com/:u:/g/personal/ych000_postech_ac_kr/EdMNBPD-1i5AsqCHNlD1F7IBAB81BjNAcTPjQ3yAPAeHQg?e=WGaRd4): `datasets`폴더로, 훈련 및 테스트에 필요한 데이터가 포함되어 있음  
- [PretrainedModel](https://postechackr-my.sharepoint.com/:u:/g/personal/ych000_postech_ac_kr/ESqHvnyemAhHj5nE1qVVTpoBkwE1-LYILpwBAungQZfAEQ?e=rRiNpz): `checkpoints`폴더로, Cosine Similarity Loss를 사용하기 위한 Pretrained Reconstruction Model Weight가 포함되어 있음
  
### DatasetInfo  
`low2high`: Original Dataset (breast, carotid, kidney, liver, thyroid)  
|--`trainA`: Low Resolution Train Dataset   
|--`trainB`: High Resolution Train Dataset  
|--`testA`: Low Resolution Test Dataset  
|--`testB`: Same to testA

## Train/Test Phase  
`sample_base.sh`: Sample Training Code
