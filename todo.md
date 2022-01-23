- 앙상블 더 적용해보기
    - 제출 파일들의 피어슨 상관계수 분석
    - 그 외 추가적인 앙상블 기법
        - 평균
        - 스태킹
        - 블렌딩

- 앙상블 모델 기준 Knowledge Distillation(?)

- Imbalanced Class
    - Data Augmentation
    - 단순 파일 복사로 개수 늘리기

- CSV 파일과 Output 분석해보기
    - 그 다음 어떻게 이를 활용할지 결정하기

- 깃헙에 올리기

model / epoch / resolution / dataset

- 전반적으로 크기 / 해상도가 작은 모델들 + Augmentation을 이용한 과적합이 효과가 좋아보인다
    - Valid F1 1을 찍고 (Train Data는 잘 예측하고)
    - Train F1이 높은 모델들 (Augmentation에 잘 대응하는 모델들)

- ConNext
- Swin S

- 실험해볼 것
    - 더 많은 모델들
    - Ensemble Resolution
        - 단일한 Res vs 여러 Res
    - Minority 들로만 Augmentation Train / 전체 Train Set으로 Validation
    - 모든 클래스 개수를 동일하게 맞춰주기
        - 기존 모델을 더 Training
        - 처음부터 새로운 모델 -> 앙상블에 추가

- CSV를 활용한 LSTM 만들기