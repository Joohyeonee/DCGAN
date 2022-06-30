#pragma once
#include "structs.h"

Ptr3D convolution(Ptr3D X, Ptr4D F, int padding, int stride);
Ptr2D calError_2D(Ptr2D E_in, Ptr2D X);
Ptr2D calWeightDiff(Ptr2D E, Ptr2D A);
Ptr2D calError_in_2D(Ptr2D E, Ptr2D W);
void updateWeight(Ptr2D &W, Ptr2D &dW);
Ptr3D calError(Ptr3D E_in, Ptr3D X);
Ptr4D calFilterDiff(Ptr4D F, Ptr3D E, Ptr3D X, int stride, int pad);
Ptr3D calError_in(Ptr3D E, Ptr4D & F, int stride, int pad);
void updateFilter(Ptr4D &F, Ptr4D &dF);
Ptr3D Leakyrelu(Ptr3D x);
Ptr2D Leakyrelu2D(Ptr2D x);
Ptr2D softmax(Ptr2D x);
Ptr2D Leakydrelu2D(Ptr2D x);
Ptr3D Leakydrelu(Ptr3D x);
double calcRMSE(Ptr2D L);
Ptr3D bNormalize(Ptr3D x);



