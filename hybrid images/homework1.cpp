#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include<iostream>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	CommandLineParser parser1(argc, argv, "{@input|youknowwho1.jfif|input image}");
	Mat image1 = imread(samples::findFile(parser1.get<String>("@input")),IMREAD_GRAYSCALE);
	Mat image2 = imread("harry2.jpg", IMREAD_GRAYSCALE);
	if (image1.empty())
	{
		cout << "could not find the image" << endl;
		cout << "usage" << argv[0] << "<input image1>" << endl;
	}
	imshow("low-ORIGIN",image1);
	if (image2.empty())
	{
		cout << "could not find the image" << endl;
		cout << "usage" << argv[0] << "<input image2>" << endl;
	}
	imshow("high-ORIGIN", image2);

//----------高通----------
	Mat padded1;
	int m1 = getOptimalDFTSize(image1.rows);
	int n1 = getOptimalDFTSize(image1.cols);
	copyMakeBorder(image1, padded1, 0, m1 - image1.rows, 0, n1 - image1.cols, BORDER_CONSTANT, Scalar::all(0));
	padded1.convertTo(padded1, CV_32FC1);

	Mat planes1[] = { Mat_<float>(padded1),Mat::zeros(padded1.size(),CV_32FC1) };
	Mat complex1;
	merge(planes1, 2, complex1);
	dft(complex1, complex1);  

	//计算magnitude然后转到对数域
	split(complex1, planes1);
	magnitude(planes1[0], planes1[1], planes1[0]);
	planes1[0] += Scalar::all(1);
	log(planes1[0], planes1[0]);
	//转换到可视图片形式
	normalize(planes1[0], planes1[0], 0, 1, NORM_MINMAX);

	Mat gaussianBlur(padded1.size(), CV_32FC2);
	float D0 = 900;//截止频率
	for (int i = 0; i < padded1.rows; i++)
	{
		float* p = gaussianBlur.ptr<float>(i);
		for (int j = 0; j < padded1.cols; j++)
		{
			float d = pow(i - padded1.rows / 2, 2) + pow(j - padded1.cols / 2, 2);
			p[2 * j] = 1 - expf(-d / D0);
			p[2 * j + 1] = 1 - expf(-d / D0);
		}
	}
	multiply(complex1, gaussianBlur, gaussianBlur);
	idft(gaussianBlur, gaussianBlur);
	split(gaussianBlur, planes1);
	


	magnitude(planes1[0], planes1[1], planes1[0]);
	normalize(planes1[0], planes1[0], 0,1, NORM_MINMAX);
	imshow("high", planes1[0]);

	//----------低通----------
	Mat padded2;
	int m2 = getOptimalDFTSize(image2.rows);
	int n2 = getOptimalDFTSize(image2.cols);
	copyMakeBorder(image2, padded2, 0, m2 - image2.rows, 0, n2 - image2.cols, BORDER_CONSTANT, Scalar::all(0));
	padded2.convertTo(padded2, CV_32FC1);
	for (int i = 0; i < padded2.rows; i++)
	{
		float* ptr = padded2.ptr<float>(i);
		for (int j = 0; j < padded2.cols; j++)
			ptr[j] *= pow(-1, i + j);
	}
	Mat planes2[] = { Mat_<float>(padded2),Mat::zeros(padded2.size(),CV_32FC1) };
	Mat complex2;
	merge(planes2, 2, complex2);
	dft(complex2, complex2);

	//计算magnitude然后转到对数域
	split(complex2, planes2);
	magnitude(planes2[0], planes2[1], planes2[0]);
	planes2[0] += Scalar::all(1);
	log(planes2[0], planes2[0]);
	//转换到可视图片形式
	normalize(planes2[0], planes2[0], 0, 1, NORM_MINMAX);

	Mat gaussianBlur2(padded2.size(), CV_32FC2);
	float D02 = 2 * 450;//截止频率
	for (int i = 0; i < padded2.rows; i++)
	{
		float* p = gaussianBlur2.ptr<float>(i);
		for (int j = 0; j < padded2.cols; j++)
		{
			float d = pow(i - padded2.rows / 2, 2) + pow(j - padded2.cols / 2, 2);
			p[2 * j] =expf(-d / D02);
			p[2 * j + 1] = expf(-d / D02);
		}
	}
	multiply(complex2, gaussianBlur2, gaussianBlur2);
	idft(gaussianBlur2, gaussianBlur2);
	split(gaussianBlur2, planes2);



	magnitude(planes2[0], planes2[1], planes2[0]);
	normalize(planes2[0], planes2[0], 0, 1, NORM_MINMAX);
	imshow("low", planes2[0]);

	//Mat dst;
	addWeighted(planes1[0], 0.5, planes2[0], 0.5,0.0 , planes2[0]);
	imshow("dst", planes2[0]);
   waitKey(0);
   return 0;
}
