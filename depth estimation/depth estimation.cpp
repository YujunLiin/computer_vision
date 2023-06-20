#include "SAD.h"
int main()
{
	Mat leftImg = imread("left.png", 0);
	Mat rightImg = imread("right.png", 0);

	Mat Disparity(leftImg.rows, leftImg.cols, CV_8U, Scalar::all(0));  //视差图

	Mat Depth(leftImg.rows, leftImg.cols, CV_8U, Scalar::all(0));      //深度图

	Sad mySAD(7, 30);
	//计算视差
	mySAD.disparity_compute(leftImg,rightImg, Disparity);
	//计算深度
	mySAD.disp2depth(Disparity,Depth);

	imshow("left image", leftImg);
	imshow("right image", rightImg);
	imshow("Disparity", Disparity);
	imshow("Depth", Depth);

	waitKey();
	return 0;
}