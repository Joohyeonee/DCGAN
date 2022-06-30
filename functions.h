#pragma once
#include "opencv2/opencv.hpp"
#include "structs.h"
#include <algorithm>
#include <experimental/filesystem>
#include <io.h>

using namespace std;
using namespace cv;


Ptr3D setImage(int dataNum, int & target);
vector<String> getImage();
Ptr3D setImage2(string path);
void feedForward(Ptr2D X, Ptr2D & Y, Ptr2D W);
Ptr2D loss(Ptr2D Y, Ptr2D T);


