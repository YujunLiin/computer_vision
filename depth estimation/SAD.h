#include<iostream>
#include<iomanip>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#define times4view 16

class Sad
{
	//窗口（类似卷积核）大小
	int windowSize;
	//视察搜索范围
	int disparity;
public:
	
	Sad() :windowSize(9), disparity(50) {}
	Sad(int win, int dis) :windowSize(win), disparity(dis) {}
	
	void disparity_compute(Mat &leftImg, Mat &rightImg,Mat dispImg);
	
	void disp2depth(Mat &disp, Mat depth);
};
void Sad::disparity_compute(Mat &leftImg, Mat &rightImg, Mat dispImg)
{
	int width = leftImg.cols;
	int height = leftImg.rows;
	//构造左右视图的窗口
	Mat leftWindow(Size(windowSize, windowSize), CV_8U, Scalar::all(0));
	Mat rightWindow(Size(windowSize, windowSize), CV_8U, Scalar::all(0));

	for (int y = 0; y < height - windowSize; y++)
	{
		for (int x = 0; x < width - windowSize; x++)
		{
			//先固定左视图窗口
			leftWindow = leftImg(Rect(x, y, windowSize, windowSize));
			//构造大小为1*disparity的图，用于计算最小视差
			Mat singleDisp(1, disparity, CV_32F, Scalar(0));
			for (int k = 0; k < disparity; k++)
			{
				int rightX = x - k;
				if (rightX > 0)
				{
					//右视图窗口
					rightWindow = rightImg(Rect(rightX, y, windowSize, windowSize));
					//计算左右窗口差值的绝对值
					Mat sub;
					absdiff(leftWindow, rightWindow, sub);
					//将差值和存入singleDisp中
					Scalar ADD = sum(sub);
					float a = ADD[0];
					singleDisp.at<float>(k) = a;
				}
			}
			//计算最小视差
			Point minLocation;
			minMaxLoc(singleDisp, NULL, NULL, &minLocation, NULL);
			dispImg.at<char>(y, x) = minLocation.x*times4view;
		}
	}
	return;
}
void Sad::disp2depth(Mat &disp,Mat depth)
{
	float f = 18;
	float baseline = 600;
	int height = disp.rows;
	int width = disp.cols;
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (disp.at<uchar>(y,x)>0)
			{
				cout << "*****" << int(disp.at<uchar>(y, x)) << endl;
				//根据公式z=(b*f)/d，b：baseline，d:视差
				depth.at<uchar>(y,x) =uchar( baseline * f / disp.at<uchar>(y, x));
			}
		}
	}
	return;
}

