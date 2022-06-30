#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include"functions.h"

using namespace cv;
using namespace std;

int main() {
	
	Ptr4D F1(96, 3, 11, 11);
	F1.random(0.05);
	
	Ptr4D F2(256, 96, 5, 5);
	F2.random(0.05);
	
	Ptr4D F3(384, 256, 3, 3);
	F3.random(0.1);
	
	Ptr4D F4(384, 384, 3, 3);
	F4.random(0.002);
	
	Ptr4D F5(256, 384, 3, 3);
	F5.random(0.05);
	
	Ptr2D W6(9216, 4096);
	W6.random(0.01);
	
	Ptr2D W7(4096, 4096);
	W7.random(0.005);
	
	Ptr2D W8(4096, 1000);
	W8.random(0.5);

	vector<String> imagePath;
	imagePath = getImage();
	int imageSize = imagePath.size();
	
	for(int ep = 0; ep < 2; ep++) {
		for(int dataNum = 0; dataNum < 10; dataNum++) {
			//¼öÁ¤Áß
		}
	}

	waitKey();
	system("pause");
	return 0;
}

