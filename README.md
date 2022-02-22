# ☘️농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

# 🔥Private LB 21위 (Score : 0.95139)
- 주최 : LG AI Research
- 주관 : 데이콘
- 목적 : "작물 환경 데이터"와 "작물 병해 이미지"를 이용해 "작물의 종류", "병해의 종류", "병해의 진행 정도"를 진단하는 AI 모델 개발
- [대회 링크](https://dacon.io/competitions/official/235870/overview/description)

# Dev Environment
- OS : Windows 11 & WSL2 (Ubuntu 20.04 LTS)
- GPU : P100 (Colab Pro, Kaggle)

# Library
- wandb 
- opencv-python-headless==4.1.2.30 
- albumentations 
- torch-summary 
- timm==0.5.4 
- einops 
- joblib 
- icecream

# 폴더 구조
```
.
├── Balanced Dataset : Data Imbalance 문제를 Oversampling으로 접근
├── CutMix : 기존 모델들 (Original Dataset)에 CutMix를 적용해 Finetuing한 모델들
├── No Aug : Augmentation 성능을 비교하기 위한 대조군
├── Original Dataset
├── SUBMIT : submission file들을 분석
├── test : SAMPLE TEST FOLDER
│   ├── 10000
│   └── 10001
└── train : SAMPLE TRAIN FOLDER
    ├── 10000
    └── 10001
```

# 접근
- 기본전략 : 다양한 이미지 모델 / 이미지 사이즈를 실험해보고, 최대한 성능을 끌어올린 단일 모델들을 추려 Ensemble하기
- [PapersWithCode](https://paperswithcode.com/)에서 SOTA 모델들을 살펴보고, timm 라이브러리를 이용해 여러 모델들을 실험
- Cutmix와 Albumentations를 이용하여 단일 모델의 성능을 최대한 끌어올리고자 함
- 한계
    - Tabular 데이터를 제대로 활용하지 못함
    - Ensemble 위주의 접근으로, predict 속도 면에서 단일 모델 위주의 접근보다 뒤처짐

## 시도한 주요 기법들
- 5-Fold Cross Validation
- ConsineAnnealingLR
    - Classifer만 5 epoch (lr : 1e-4)
    - 이후 전체 모델 Training (lr : 1e-6)
- Image Augmentation (albumentations)
```python
data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
        ], p=1.),

    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(),
        ToTensorV2()], p=1.)
}
```
- Cutmix
- Ensemble (Soft Voting)

## 시도했으나 채택되지 못한 기법들
- TTA
    - TTA vs 더 많은 모델 Ensemble 중에서 고민하다, 최종적으로는 후자를 선택. TTA는 사용되지 않음.
- Oversampling
    - EDA를 통해 데이터를 살펴보았을 때, 특정 클래스의 수가 상대적으로 "매우" 부족함을 확인
    - 단순 Oversampling 기법을 통해서 데이터의 수를 맞춰주려고 시도함.
    ![Oversampling Image](https://i.imgur.com/uBPlTpV.png)
    - 그러나 생각보다 유효한 효과는 나타나지 않음. 이후 Data의 개수를 늘리는 Oversampling보다는 강하고 다양한 Augmentation을 통해서 문제를 극복하려는 전략을 취함.

# 모델 테스트
- 경험적으로, Train 데이터에 과적합될수록 오히려 성능이 LB 스코어가 잘나오는 것을 확인하고, 대부분의 모델을 (Valid F1이 1이 될때까지) 과적합시켜서 테스트함. 
- TTA 모델의 경우, 너무 많은 Augmentation은 오히려 성능 하락을 유발. 0<sup>0</sup>, 90<sup>0</sup>, 180<sup>0</sup>, 270<sup>0</sup> 네가지의 기본 회전만으로 TTA Augmentation을 적용함.
- 전반적으로 CNN 모델보다는 Transformer 계열의 모델들이, 이미지 크기(Resolution)가 큰 모델들보다는 작은 모델들이 좋은 성능을 보여줌.
- CutMix는 성능 향상에 매우 좋은 Augmentation. CutMix를 적용한 이후 대부분의 모델들이 이전보다 높은 성능을 보여줌.
- 다음 표는 Valid F1 기준으로 모델들을 내림차순 정렬한 것이다.

| Model | CutMix | TTA | Resolution | Param | Epoch | Train F1 | Valid F1 | Public LB | Note |
|---|---|---|---|---|---|---|---|---|---|
| ENSEMBLE #13 | Y | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.949678674 | original_finecutmix_effnetv2m_swin_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE   #12 (11 TTA) | Y | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.948370833 | finecutmix_TTA_effnetv2m_swin_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE #8 (9) |  | Y | ENSEMBLE |  | ENSEMBLE |  |  | 0.948030037 | effnetv2m(62)_swin(57)_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224_TTA |
| ENSEMBLE   #7 (9) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.947822635 | effnetv2m(62)_swin(57)_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE #6 (9) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.947634928 | effnetv2m(3)_swin(3)_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE   #3 (9) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.946883266 | effnetv2m(3)_swin(3)_deit_b4ns512_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE #11 | Y | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.946642688 | finecutmix_effnetv2m_swin_deit_b4ns512(160)_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| ENSEMBLE   #4 (9) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.946176973 | effnetv2m(3)_swin(3)_deit_b4ns512_coatmini224_beit224in22k_cait224_[B]swinS224_convnextS224 |
| ENSEMBLE #9 (8) | Y | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.946096307 | effnetv2m_swin_deit_coatmini224_beit224in22k_cait224_swinS224_convnextS224 |
| DeiT | Y | N | 384 | 85M | 201 | 0.8811 | 1 | 0.945885493 | w |
| ENSEMBLE #5 (5) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.94480485 | coatmini224_beit224in22k_cait224_[B]swinS224_convnextS224 |
| ENSEMBLE   #2 (6) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.944121324 | effnetv2m(3)_swinB(3)_deit_b4ns512_coatmini224 |
| cait_s24_224 | Y | N | 224 | 46M | 40 | 0.7338 | 1 | 0.942474188 |  |
| Swin-B | Y | N | 384 | 86M | 125 | 0.8078 | 1 | 0.941872109 |  |
| Swin-S | Y | N | 224 | 48M | 56 | 0.8685 | 1 | 0.941198225 | swin_small_patch4_window7_224 |
| BeiT   22k | Y | N | 224 | 85M | 374 | 0.8792 | 1 | 0.941053769 | beit_base_patch16_224_in22k |
| ENSEMBLE #1 (4) |  | N | ENSEMBLE |  | ENSEMBLE |  |  | 0.939868495 | effnetv2m(3)_swinB(3)_deit_b4ns(50) |
| coat_mini | Y | N | 224 | 16M | 368 | 0.7699 | 1 | 0.937212932 |  |
| BeiT 22k | N | Y | 224 | 85M | 55 |  | 1 | 0.935879431 | beit_base_patch16_224_in22k |
| tf_efficientnetv2_m | Y | N | 384 | 52M | 62 | 0.7111 | 1 | 0.93544119 | tf_efficientnetv2_m |
| BeiT 22k | N | N | 224 | 85M | 55 | 0.9918 | 1 | 0.935041547 | beit_base_patch16_224_in22k |
| Swin-S | N | N | 224 | 48M | 56 | 0.9968 | 1 | 0.934686009 | swin_small_patch4_window7_224 |
| cait_s24_224 | N | N | 224 | 46M | 40 | 0.9913 | 1 | 0.932752062 |  |
| ConvNext-S | N | N | 224 | 49M | 73 | 0.9967 | 1 | 0.932625441 | convnext_small |
| Swin-B | N | N | 384 | 86M | 57 | 0.9959 | 1 | 0.932183115 | swin_base_patch4_window12_384 |
| coat_mini | N | N | 224 | 16M | 45 | 0.9968 | 1 | 0.931627762 |  |
| ConvNext-S | Y | N | 224 | 49M | 405 |  | 1 | 0.931021272 | convnext_small |
| DeiT | N | N | 384 | 85M | 13   (finetune) | 0.9878 | 0.9952 | 0.927680222 | deit_base_distilled_patch16_384 |
| effnet_b4_ns | N | N | 512 | 17M | 71 | 0.9893 | 1 | 0.927216641 | tf_efficientnet_b4_ns |
| effnet_b4_ns | N | N | 384 | 17M | 82 | 0.9896 | 1 | 0.925359356 | tf_efficientnet_b4_ns |
| tf_efficientnetv2_m | N | N | 384 | 52M | 62 | 0.9926 | 1 | 0.925318381 | tf_efficientnetv2_m |
| effnet_b6_ns | N | N | 512 | 　 | 83 | 0.994 | 1 | 0.908087196 | tf_efficientnet_b6_ns |

# 개선해야 할 점 | 시도해볼만 한 것들 (From 1st ~ 10th Solution)
- Tabular 데이터의 활용
    - LSTM, CatBoost 등
- bbox등 추가 데이터를 활용한 Augmentation도 가능했다. 
    - ex) CutMix할 때 두 이미지의 bbox를 무조건 포함하도록 하기.
- Cross Entropy Loss를 사용했는데, 이러한 Imbalanced Data일 경우 Focal Loss를 시도해볼 수 있겠다.
- 추가적인 Regularization
    - Label Smoothing
