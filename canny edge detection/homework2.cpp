#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>   //添加Surf
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;     //添加命名空间

int main(int argc, char** argv)
{
	Mat src1 = imread("Andrew-3.jpg");
	Mat src2 = imread("Andrew-4.jpg");
	//计算Surf特征点
	Ptr<SURF> surfdect = SURF::create(1000);
	vector<KeyPoint> keypoint_src1;
	vector<KeyPoint> keypoint_src2;
	surfdect->detect(src1, keypoint_src1);
	surfdect->detect(src2, keypoint_src2);
	//计算描述符（特征向量）
	Mat descriptor_src1;
	Mat descriptor_src2;
	surfdect->compute(src1, keypoint_src1, descriptor_src1);
	surfdect->compute(src2, keypoint_src2, descriptor_src2);
	//使用FLANN算法匹配描述符向量
	vector<DMatch> dMatch;
	Ptr<FlannBasedMatcher> flannMatcher = FlannBasedMatcher::create();
	flannMatcher->match(descriptor_src1, descriptor_src2, dMatch);
	double max_dist = 0, min_dist = 100;
	//快速计算关键点之间的最大和最小距离
	for (int i = 0; i < descriptor_src1.rows; i++)
	{
		double dist = dMatch[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	//输出距离信息
	printf("Max dist : %f \n", max_dist);
	printf("Min dist : %f \n", min_dist);

	//储存符合条件的匹配结果（dist < 2 * min_dist）
	vector<DMatch> good_match;
	for (int i = 0; i < descriptor_src1.rows; i++)
	{
		if (dMatch[i].distance < 2 * min_dist)
		{
			good_match.push_back(dMatch[i]);
		}
	}

	//绘制符合条件的匹配点
	Mat imgMatch;
	drawMatches(src1, keypoint_src1, src2, keypoint_src2, good_match, imgMatch,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//输出相关点匹配信息
	for (int i = 0; i < good_match.size(); i++)
	{
		printf("> 符合条件的匹配点 [%d] 特征点1：%d --- 特征点2： %d  \n",
			i, good_match[i].queryIdx, good_match[i].trainIdx);
	}
	//显示匹配结果
	imshow("FLANN Match", imgMatch);

	waitKey(0);
	return 0;
}