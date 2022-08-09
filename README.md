

#GAN
---

#DCGAN
![링크](https://arxiv.org/pdf/1511.06434.pdf)

0. functions
 - get_data() : 입출력 데이터를 담는 Tensor 생성
 - randn() : 가중치 초기화
 - simple_network() : y = Wx + b
 - loss_fn() : MSE 계산
 - cuda() : GPU에서 동작하는 Tensor 객체로 복사
 - optim.optimizer() : optimizer 적용
 - optimizer.step() : 가중치 갱신

1. terminology
 - Discriminator : fake 이미지와 real 이미지를 구별한다.
 - Generator : uniform random noise로부터 fake 이미지를 생성한다.

2. Architecture
 - Generator<br/>
 >   z(noise)   : 1 x 100 / uniform random noise vector
 >
 >       |      f(project and reshape / )
 >
 >      f(z)    : 4 x 4 x 1024
 >
 >       |      T(transpose convolution / kernel size = )   
 >
 >     T(f(z))   : 8 x 8 x 512 = t1
 >
 >       |      T(transpose convolution / kernel size = )
 >
 >     T(t1)    : 16 x 16 x 256 = t2
 >
 >       |      T(transpose convolution / kernel size = )
 >
 >     T(t2)    : 32 x 32 x 128 = t3
 >
 >       |      T(transpose convolution / kernel size =  )
 >
 >     T(t3)    : 64 x 64 x 3 = G

 - Discriminator<br/>
 >       X      : 64 x 64 x 3 
 >
 >       |      H(Convolution / kernel size = 4 x 4 x 3 x 64, stride = 2, padding = 1)
 >
 >      H(X)    : 32 x 32 x 128 = h1
 >
 >       |      H(Convolution / kernel size = 4 x 4 x 64 x 128, stride = 2, padding = 1)
 >
 >      H(h1)   : 16 x 16 x 256 = h2
 >
 >       |      H(Convolution  / kernel size = 4 x 4 x 128 x 256, stride = 2, padding = 1)
 >
 >      H(h2)   : 8 x 8 x 512 = h3
 >
 >       |      H(Convolution  / kernel size = 4 x 4 x 256 x 512, stride = 2, padding = 1)
 >
 >      H(h3)   : 4 x 4 x 1024 = h4
 >
 >       |      H(Convolution  / kernel size =  4 x 4 x 512 x 1, stride = 2, padding = 1)
 >
 >      H(h4)   : 1 x 1 x 1 = D  

 3. Loss Function
  - Binary Cross Entropy Loss : 0(fake) 또는 1(real)로 구분

---
3. DCGAN의 차이점
 - 기존 GAN의 Fully-Connected Layer, Pooling Layer를 Convolution Layer로 대체(지역적 정보 손실 방지)
 - Batch Normalization 도입 : 입력 데이터의 평균, 분산을 조정하여 학습 효과 증대
 - 학습 검증을 위한 방법 도입 : 잠재 공간에 의미 있는 단위(침실 이미지를 학습할 시 창문, 침대 등의 단위)가 존재하는지 여부 확인 -> randomized vector의 값을 조금씩 변경하여 확인
 - 수많은 학습을 통한 세부 조건 확립 : learning rate, weight initialization, activation function 등
 ---
 
 #GPEN(Gan Prior Embedded Network for Blind FAce Restoration in the Wild) / embed a GAN prior network into a DNN
 0. Introduction
  - face restoration의 발전에도 불구하고 blind face resolution은 알 수 없는 degradation 때문에 문제가 있었음
  - Hifacegan에서는 얼굴 디테일을 복원하기 위한 CSR(Collaborative Suppresion and Replenishment) 접근을 제안했지만 현실 세계의 저화질 이미지를 다루는데 실패했음
  - cGAN 기반 Pix2Pix, Pix2PixHD는 input부터 output까지 direct mapping을 학습 -> 현실적인 결과를 달성하지만 과하게 smooth한 이미지를 생성하는 경향이 있음
  - 이 논문에서는 GAN과 DNN의 장점을 통합하여 고화질 이미지 생성에 GAN을 pre-train하고 decoder로 DNN을 사용한
  - DNN이 degraded 이미지를 원하는 latent space에 mapping 하는 것을 학습하여 GAN이 고화질 이미지로 재생산하는 동안, GAN prior embedded DNN은 저화질-고화질 이미지 세트로 fine-tunning됨
  - U-shaped DNN에서 깊은 feature는 global face reproduction을 위한 latent code를 만들고, 얕은 feature는 local face detail을 만드는 noise로 사용됨
  - 학습된 모델은 심각하게 열화된 이미지에서  과하게 부드럽지 않으면서도 사진같은 디테일의 고화질 얼굴을 복원할 수 있음

1. Method
 - source domain과 target domain은 같고, 얼굴 생성을 학습한 GAN을 DNN에 포함시키고, GAN과 DNN을 같이 fine-tunning하여 latent code와 noise input이 다른 network layer의 열화된 이미지로부터 잘 생성되도록 함
 - 1st part of GPEN : 열화된 입력 이미지 x를 GAN의 latent space 속의 latent code z로 매핑해주는 CNN encoder
 - 2nd part of GPEN : latent code z를 w로 projection하여 GAN block으로 전파(skip connection을 위한 추가적인 noise input)
 - 3rd part of GPEN : GAN의 generator가 G(z) -> y의 매커니즘을 통해 고화질 이미지 생성(1:1 매핑)
 - GAN block의 구조는 여러 option이 있지만 StyleGAN v2의 구조 채택 / StyleGAN은 각 GAN block에 두 개의 다른 noise input을 요구함
 - GAN block의 개수는 U-shapped DNN에서 추출된 skipped feature map의 개수와 동일
 - GAN network를 U-shaped GPEN에 포함하도록 하기 위해서, StyleGAN과 다르게 noise input은 모든 GAN block에 재사용됨(경험적으로 발견)
 - latent code z와 noise는 각각 global한 얼굴 구조를 control하는 deeper feature과 local detail을 control하는 얕은 layer의 output으로 교체됨
 - GPEN의 입력으로 들어가기 전에 bilinear interpolator을 사용하여 원하는 크기로 resize됨

2. Training Strategy
 - Loss : adversarial loss, content loss, feature matching loss 3가지를 모두 사용함
 > feature mapping loss : perceptual loss와 비슷하지만 pre-trained VGG network를 사용하는 대신 discriminator 기반
 > input image에 대한 discriminator output과 SR 이미지에 대한 discriminator의 output의 norm의 평균을 minimize
 - GAN train 시 FFHQ dataset 사용(1024 x 1024), 다른 SOTA 모델과 비교 시 (evaluation 시) CelebA-HQ dataset 사용
