#include "functions.h"

Ptr3D setImage(int dataNum, int & target) 
{
	Mat img; 
	Mat re_img;
	Mat rgb_img[3];

	string path = "E:\\archive\\imagenet-mini\\train\\";
	vector<string> dir;
	for (auto & p : experimental::filesystem::directory_iterator(path))
	{
		dir.push_back(p.path().string());
	}
	string* dirName;
	dirName = new string[1000];
	size_t pos = dir[0].rfind("n");

	vector<string> tmp2;
	vector<vector<string>> filelist;

	for (int i = 0; i < 1000; i++) {
		dirName[i] = dir[i].substr(pos, pos + 8);
		string dirname = dirName[i];
		string path2 = path + dirname + "\\" + dirname + "_" + to_string(dataNum) + ".JPEG";
		ifstream f(path2);
		if (f.is_open()) {
			img = cv::imread(path2, cv::IMREAD_COLOR);

			target = i;
			break;
		}
	}

	resize(img, re_img, Size(256, 256));
	split(re_img, rgb_img);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> distribution(0,29);

	int startR = distribution(gen);
	int startC = distribution(gen);

	Ptr3D tmp(3, 227, 227);
	int Nrow = startR + 227;
	int Ncol = startC + 227;
	int Ndep = 3;

	for (int i = Ndep - 1; i >= 0; i--) {
		for (int j = startR; j < Nrow; j++) {
			for (int k = startC; k < Ncol; k++) {
				tmp[i][j - startR][k - startC] = rgb_img[i].at<uchar>(j, k) / 255.f;
			}
		}
	}
	return tmp;
}

Ptr3D setImage2(string path)
{
	Mat img;
	Mat re_img;
	Mat rgb_img[3];
	img = cv::imread(path, cv::IMREAD_COLOR);

	Ptr3D tmp(3, 227, 227);
	resize(img, re_img, Size(256, 256));
	split(re_img, rgb_img);

	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> distribution(0, 28);

	int startR = distribution(gen);
	int startC = distribution(gen);
	int Nrow = startR + 227;
	int Ncol = startC + 227;
	int Ndep = 3;

	for (int i = Ndep - 1; i >= 0; i--) {
		for (int j = startR; j < Nrow; j++) {
			for (int k = startC; k < Ncol; k++) {
				tmp[i][j - startR][k - startC] = rgb_img[i].at<uchar>(j, k) / 255.f;
			}
		}
	}
	return tmp;
}

vector<String> getImage() {
	vector<String> filenames;
	string path = "E:\\archive\\imagenet-mini\\train\\*.JPEG";
	ifstream f(path);
	string f_path = "E:\\archive\\imagenet-mini\\train\\";

	glob(path, filenames, true);

	if (filenames.size() == 0) {
		cout << "이미지가 존재하지 않습니다.\n" << endl;
	}

	return filenames;
}

Ptr2D setTarget(int target)
{
	Ptr2D tmp(1, 1000);
	for (int i = 0; i < 1000; i++) {
		if (i == target) tmp[0][i] = 1;
	}
	return tmp;
}


void feedForward(Ptr2D X, Ptr2D & Y, Ptr2D W) 
{
	int I = Y.width;
	int J = X.width;
	Ptr2D tmp(1,I);
	for (int i = 0; i < I; i++) {
		for (int j = 0; j < J; j++) {
			tmp[0][i] += X[0][j] * W[j][i];
		}
	}
	Y = tmp;
}

Ptr2D loss(Ptr2D Y, Ptr2D T) 
{
	Ptr2D tmp(1, Y.width);
	for (int i = 0; i < Y.width; i++) {
		tmp[0][i] = Y[0][i] - T[0][i];
	}
	return tmp;
}

