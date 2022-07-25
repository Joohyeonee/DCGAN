# DCGAN
---
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
