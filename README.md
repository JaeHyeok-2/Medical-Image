## `Medical Image`

`To do List` 
- [x] SVS File을 CLAM [1] 의 Heatmap Visualization Code를 통해 Tumor, Normal의 Attention Heatmap JPEG image 얻기
- [x] HEATMAP 부분에서 Tumor 부분(RED COLOR) 부분을 선택
  - [x] PATCH의 개인적으로 PIXEL 값을 계산해서 누끼(Pick-up)를 따기
  - [x] 누끼를 딴 부분(JPEG)과 SVS 파일을 Matching해서 Patch Size에 맞게 Dataset 생성
  - [x] 생성된 Dataset으로 Model 훈련 
 
 
 ![alt text](TumorZoom.gif)
 
  
### segmentation.ipynb
해당 코드를 통해서 CLAM의 VISUALIZATION HEATMAP 결과 이미지를 Pick-up
결과 저장은 코드내부에서 변경할 수 있음

### make_patch.ipynb
위의 segmentation.ipynb의 결과물과 SVS File을 맵핑해서 원하는 크기의 패치를 만들어 냄 
데이터 저장경로는 내부코드에서 변경가능 


### train.py, dataset.py
위의 2개의 파일을 돌린 후 나온 데이터셋을 Model에 훈련시키기 위한 Dataset으로 변경
learning_rate, epochs, labels, batch_size, model_save_path, pretrained 

**learning_rate** : Model의 Learning rate를 설정 (SGD) Default = 0.001
**epochs** : 훈련시킬 Epochs 수
**labels** : torchvisions.models에 존재하는 Resnet18 모델을 사용, class수 결정 $\rightarrow$ 
            해당 Classification에서는 ALK+ , ALK- Tumor를 분류하는 Task로 labels =2
            
**model_save_path** : ResNet18 훈련 후 가중치 저장 할 곳
**pretrained** : Pretrained Model을 불러올 곳

<!-- 첫번째 줄 -->
<p align="center">
  <img src="IMAGE/0.jpg" width="30%" />
  <img src="IMAGE/1.jpg" width="30%" />
  <img src="IMAGE/2.jpg" width="30%" />
</p>

<!-- 두번째 줄 -->
<p align="center">
  <img src="IMAGE/3.jpg" width="30%" />
  <img src="IMAGE/4.jpg" width="30%" />
  <img src="IMAGE/5.jpg" width="30%" />
</p>
### `Reference`




[1]: https://github.com/mahmoodlab/CLAM
