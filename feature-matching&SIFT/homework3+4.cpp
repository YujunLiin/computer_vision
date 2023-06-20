#include <windows.h>
#include <iostream>
#include <opencv/cv.h>
//#include <opencv/cxcore.h>
//#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "sift.h"


using namespace std;
using namespace cv;


int main(int argc, char **argv)
{
	cv::Mat img1 = imread("b1-2.jpg");
	cv::Mat img2 = imread("b3-2.jpg");

	if (img1.empty())
	{
		cout << "d1.jpg open error! " << endl;
		getchar();
		return -1;
	}
	if (img2.empty())
	{
		cout << "d3.jpg open error! " << endl;
		getchar();
		return -1;
	}

	if (img1.channels() != 3) return -2;
	if (img2.channels() != 3) return -2;

	vector<Keypoint> features1;
	vector<Keypoint> features2;

	Sift(img1, features1, 1.6);                           //【1】SIFT特征点检测和特征点描述
	Sift(img2, features2, 1.6);

	DrawKeyPoints(img1, features1);                       //【2】画出关键点(特征点)
	cv::imshow("key points 1", img1);

	DrawSiftFeatures(img1, features1);                    //【3】画出SIFT特征点
	cv::imshow("features 1", img1);

	DrawKeyPoints(img2, features2);                       
	cv::imshow("key points 2", img2);

	DrawSiftFeatures(img2, features2);                   
	cv::imshow("features 2", img2);

	int row1 = img1.rows, col1 = img1.cols, row2 = img2.rows, col2 = img2.cols;
	int row = max(row1, row2), col = col1 + col2 + 100;
	Mat img = Mat(row, col, CV_8UC3);
	for (int r = 0; r < row1; r++) {
		for (int c = 0; c < col1; c++) {
			img.ptr<uchar>(r)[c * 3 + 0] = img1.ptr<uchar>(r)[c * 3 + 0];
			img.ptr<uchar>(r)[c * 3 + 1] = img1.ptr<uchar>(r)[c * 3 + 1];
			img.ptr<uchar>(r)[c * 3 + 2] = img1.ptr<uchar>(r)[c * 3 + 2];
		}
	}
	for (int r = 0; r < row2; r++) {
		for (int c = 0; c < col2; c++) {
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 0] = img2.ptr<uchar>(r)[c * 3 + 0];
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 1] = img2.ptr<uchar>(r)[c * 3 + 1];
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 2] = img2.ptr<uchar>(r)[c * 3 + 2];
		}
	}
	vector<matchPoint> match = compute_match(features1, features2, 0.003);
	int count = 1;
	for (matchPoint mp : match) {
		cout << count++ << endl;
		line(img, mp.p1, Point2i(mp.p2.x + col1 + 100, mp.p2.y), Scalar(160, 20, 100), 2, 8, 0);
	}
	cout << "!" << endl;
	cv::imshow("1", img);
	cv::waitKey(0);

	cv::waitKey();

	return 0;
}
