# DCGAN
---
![링크](https://arxiv.org/pdf/1511.06434.pdf)

1. terminology
 - Discriminator : fake 이미지와 real 이미지를 구별한다.
 - Generator : uniform random noise로부터 fake 이미지를 생성한다.

2. Architecture
 - Generator<br/>
 >   z(noise)   : 1 x 100 / uniform random noise vector
 >
 >       |      f(project and reshape / )
 >
 >      f(z)    : 4 x 4 x 512
 >
 >       |      T(transpose convolution / kernel size = )   
 >
 >     T(f(z))   : 8 x 8 x 256 = t1
 >
 >       |      T(transpose convolution / kernel size = )
 >
 >     T(t1)    : 16 x 16 x 128 = t2
 >
 >       |      T(transpose convolution / kernel size = )
 >
 >     T(t2)    : 32 x 32 x 64 = t3
 >
 >       |      T(transpose convolution / kernel size =  )
 >
 >     T(t3)    : 64 x 64 x 3 = G

 - Discriminator<br/>
 >       X      : 64 x 64 x 3 
 >
 >       |      H(Convolution / kernel size = 4 x 4 x 3 x 64, stride = 2, padding = 1)
 >
 >      H(X)    : 32 x 32 x 64 = h1
 >
 >       |      H(Convolution / kernel size = 4 x 4 x 64 x 128, stride = 2, padding = 1)
 >
 >      H(h1)   : 16 x 16 x 128 = h2
 >
 >       |      H(Convolution  / kernel size = 4 x 4 x 128 x 256, stride = 2, padding = 1)
 >
 >      H(h2)   : 8 x 8 x 256 = h3
 >
 >       |      H(Convolution  / kernel size = 4 x 4 x 256 x 512, stride = 2, padding = 1)
 >
 >      H(h3)   : 4 x 4 x 512 = h4
 >
 >       |      H(Convolution  / kernel size =  4 x 4 x 512 x 1, stride = 2, padding = 1)
 >
 >      H(h4)   : 1 x 1 x 1 = D  

 3. Loss Function
  - Binary Cross Entropy Loss : 0(fake) 또는 1(real)로 구분

---
