//Stereo Match By ZNCC
//参考https://www.cnblogs.com/hxjbc/p/6432378.html
//效果比较https://www.cnblogs.com/hxjbc/p/6432378.html

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream> 
#include <string>  
#include <stdio.h>
#include<opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

int64 t1;  
int64 t2;  

void timebegin()  
{  
    t1 = getTickCount();  
}  

void timeend(string str)  
{  
    t2 = getTickCount();  
    cout << str << " is "<< (t2 - t1)*1000/getTickFrequency() << "ms" << endl;  
}  

void MatDataNormal(const Mat &src, Mat &dst)
{
    normalize(src, dst, 255, 0, NORM_MINMAX );
    dst.convertTo(dst, CV_8UC1);
}

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

    return res;
}

//左图加上视差等于有右图
void SteroMatchByZNCC(Mat &disparity, const Mat &leftimg, const Mat  &rightimg, 
    const int MaxDisparity, const  int winsize)
{
    int row = leftimg.rows;
    int col = leftimg.cols;

    int w = (winsize-1)/2;
    int rowrange = row - w;
    int colrange = col - w - MaxDisparity;

    for (int i = w; i < rowrange; ++i)
    {
        int *ptr = disparity.ptr<int>(i);
        for (int j = w; j < colrange; ++j)
        {
            //Rect rightrect;
            Mat rightwin = rightimg(Range(i - w,i + w + 1),Range(j - w,j + w + 1)); 
            float preZncc =  0.0;
            int disp = 0;
            for (int d = j; d < j + MaxDisparity; ++d)
            {
                //Rect leftrect;
                Mat leftwin = leftimg(Range(i - w,i + w + 1),Range(d - w,d + w + 1));
                float curZncc = calZnccValue(leftwin, rightwin);
                if(curZncc > preZncc)
                {
                    preZncc = curZncc;
                    disp = d-j;
                }
            }
            *(ptr + j) = disp;
        }
    }
    return;
}

int main()
{
    Mat leftimg = imread("left.jpg",0);   
    Mat rightimg = imread("right.jpg",0); 
    if (leftimg.channels() == 3 && rightimg.channels() == 3)
    {
        cvtColor(leftimg, leftimg, CV_BGR2GRAY);
        cvtColor(rightimg, rightimg, CV_BGR2GRAY);
    }

    float scale = 0.5;
    int row = leftimg.rows * scale;
    int col = leftimg.cols * scale;
    resize(leftimg, leftimg, Size( col, row));
    resize(rightimg,rightimg, Size(col, row));
    Mat lastdisp = Mat ::zeros(row,col, CV_32S);
    int MaxDisparity = 16;
    int winsize = 7;

    timebegin();
    SteroMatchByZNCC(lastdisp,leftimg, rightimg, MaxDisparity, winsize);
    timeend("time ");

    MatDataNormal(lastdisp, lastdisp);
    namedWindow("left", 0);
    namedWindow("right", 0);
    namedWindow("lastdisp",0);
    imshow("left", leftimg);
    imshow("right", rightimg);
    imshow("lastdisp",lastdisp);

    string strsave = "result_";
    imwrite(strsave +"lastdisp.jpg",lastdisp);
    waitKey(0);
    return 0;
}