#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream> 
#include <string>  
#include <stdio.h>
#include<opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

float calZnccValue(const Mat &src1, const Mat &src2)
{
    Mat m1, m2;

    src1.convertTo(m1, CV_32F);
    src2.convertTo(m2, CV_32F);

    cv::Scalar     varMean, varMean2;
	cv::Scalar     varDev, varDev2;
	cv::meanStdDev(m1, varMean, varDev);
	cv::meanStdDev(m2, varMean2, varDev2);

    float mean1 = varMean.val[0];
    float var1 = varDev.val[0];
    float mean2 = varMean2.val[0];
    float var2 = varDev2.val[0];

    Mat part1 = m1 - mean1;
    Mat part2 = m2 - mean2;
    Mat part3 = part1.mul(part2);
    if(var1 < 0.000001)
        return 0.0;
    if(var2 < 0.000001)
        return 0.0;  
    float res = (float)mean(part3).val[0]/(var1*var2);
    cout << res << endl;
    return res;
}


int main()
{
    float m[2][2] = { 1, 2, 3, 4};
    float m2[2][2] = { 1, 2, 2, 4};

    Mat t1(2,2,CV_32F,m);
    Mat t2(2,2,CV_32F,m);
    Mat t3(2,2,CV_32F,m2);

    calZnccValue(t1, t2);
    calZnccValue(t1, t3);

    // int mat[2][2] = { 1, 2, 2, 3};
    // Mat a(2,2,CV_32S,mat);
    // cout << a << endl;
    // a.convertTo(a, CV_32F);
    // Mat b = a.clone();
    // Mat c = Mat::ones(2,2, CV_32F);

    // cout << a << endl;
    // cv::Scalar     varMean;
	// cv::Scalar     varDev;
	// cv::meanStdDev(a, varMean, varDev);
	// float       m = varMean.val[0];
	// float       s = varDev.val[0];

    // b = b - m;
    // cout << b << endl;
    // Mat d = a.mul(c);
    // Mat e = a*c;
    // cv::Scalar dm = mean(d);
    // float z = dm.val[0];
    // cout << d << endl;
    // cout << e << endl;
    return 0;
}