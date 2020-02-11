//Stereo Match By SAD
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


float sadvalue(const Mat &src1, const Mat &src2)
{
    Mat  matdiff = cv::abs(src1 -src2);
    int  saddiff = cv::sum(matdiff)[0];
    return saddiff;
}

float GetMinSadIndex(std::vector<float> &sad)
{
    float minsad = sad[0];
    int index = 0;
    int len = sad.size();
    for (int i = 1; i < len; ++i)
    {
        if (sad[i] < minsad)
        {
            minsad = sad[i];
            index = i;
        }
    }
    return index;
}

void MatDataNormal(const Mat &src, Mat &dst)
{
    normalize(src, dst, 255, 0, NORM_MINMAX );
    dst.convertTo(dst, CV_8UC1);
}


void GetPointDepthRight(Mat &disparity, const Mat &leftimg, const Mat  &rightimg, 
    const int MaxDisparity, const  int winsize)
{
    int row = leftimg.rows;
    int col = leftimg.cols;
    if (leftimg.channels() == 3 && rightimg.channels() == 3)
    {
        cvtColor(leftimg, leftimg, CV_BGR2GRAY);
        cvtColor(rightimg, rightimg, CV_BGR2GRAY);
    }

    //Mat disparity = Mat ::zeros(row,col, CV_32S);
    int w = winsize;
    int rowrange = row - w;
    int colrange = col - w - MaxDisparity;

    for (int i = w; i < rowrange; ++i)
    {
        int *ptr = disparity.ptr<int>(i);
        for (int j = w; j < colrange; ++j)
        {
            //Rect rightrect;
            Mat rightwin = rightimg(Range(i - w,i + w + 1),Range(j - w,j + w + 1)); 
            std::vector<float> sad(MaxDisparity);
            for (int d = j; d < j + MaxDisparity; ++d)
            {
                //Rect leftrect;
                Mat leftwin = leftimg(Range(i - w,i + w + 1),Range(d - w,d + w + 1));
                sad[d - j] = sadvalue(leftwin, rightwin);
            }
            *(ptr + j) = GetMinSadIndex(sad);
        }
    }
}

void GetPointDepthLeft(Mat &disparity, const  Mat &leftimg, const Mat  &rightimg, 
    const int MaxDisparity, const  int winsize)
{
    int row = leftimg.rows;
    int col = leftimg.cols;
    if (leftimg.channels() == 3 && rightimg.channels() == 3)
    {
        cvtColor(leftimg, leftimg, CV_BGR2GRAY);
        cvtColor(rightimg, rightimg, CV_BGR2GRAY);
    }

    //Mat disparity = Mat ::zeros(row,col, CV_32S);
    int w = winsize;
    int rowrange = row - w;
    int colrange = col - w;

    for (int i = w; i < rowrange; ++i)
    {
        int *ptr = disparity.ptr<int>(i);
        for (int j = MaxDisparity + w; j < colrange; ++j)
        {
            //Rect leftrect;
            Mat leftwin = leftimg(Range(i - w,i + w + 1),Range(j - w,j + w + 1)); 
            std::vector<float> sad(MaxDisparity);
            for (int d = j; d >  j -  MaxDisparity; --d)
            {
                //Rect rightrect;
                Mat rightwin = rightimg(Range(i - w,i + w + 1),Range(d - w,d + w + 1));
                sad[j - d] = sadvalue(leftwin, rightwin);
            }
            *(ptr + j) = GetMinSadIndex(sad);
        }
    }
}

//(Left-Right Consistency (LRC)
void CrossCheckDiaparity(const Mat &leftdisp, const Mat &rightdisp, Mat &lastdisp, 
    const int MaxDisparity, const int winsize)
{
    int row = leftdisp.rows;
    int col = rightdisp.cols;
    int w = winsize;
    int rowrange = row - w;
    int colrange = col - MaxDisparity - w;
    int diffthreshold = 2;
    for (int i = w; i < row -w; ++i)
    {
        const int *ptrleft = leftdisp.ptr<int>(i);
        const int *ptrright = rightdisp.ptr<int>(i);
        int *ptrdisp = lastdisp.ptr<int>(i);
        for (int j = MaxDisparity + w; j < col - MaxDisparity - w; ++j)
        {
            int leftvalue = *(ptrleft + j);
            int rightvalue = *(ptrright + j - leftvalue );
            int diff = abs(leftvalue - rightvalue);
            if (diff > diffthreshold)
            {
                *(ptrdisp + j) = 0;
            }else
            {
                *(ptrdisp + j) = leftvalue;
            }
        }
    }

}


int main()
{
    // Mat leftimg = imread("aloeL.jpg",0);   
    // Mat rightimg = imread("aloeR.jpg",0); 
    Mat leftimg = imread("left.png",0);   
    Mat rightimg = imread("right.png",0); 
    if (leftimg.channels() == 3 && rightimg.channels() == 3)
    {
        cvtColor(leftimg, leftimg, CV_BGR2GRAY);
        cvtColor(rightimg, rightimg, CV_BGR2GRAY);
    }

    float scale = 1;
    int row = leftimg.rows * scale;
    int col = leftimg.cols * scale;
    resize(leftimg, leftimg, Size( col, row));
    resize(rightimg,rightimg, Size(col, row));
    Mat depthleft = Mat ::zeros(row,col, CV_32S);
    Mat depthright = Mat ::zeros(row,col, CV_32S);
    Mat lastdisp = Mat ::zeros(row,col, CV_32S);
    int MaxDisparity = 60 * scale;
    int winsize = 31*scale;

    timebegin();
    GetPointDepthLeft(depthleft, leftimg, rightimg, MaxDisparity,  winsize);
    GetPointDepthRight(depthright, leftimg, rightimg, MaxDisparity,  winsize);
    CrossCheckDiaparity(depthleft,depthright, lastdisp, MaxDisparity, winsize);
    timeend("time ");

    MatDataNormal(depthleft,depthleft);
    MatDataNormal(depthright, depthright);
    MatDataNormal(lastdisp, lastdisp);
    namedWindow("left", 0);
    namedWindow("right", 0);
    namedWindow("depthleft", 0);
    namedWindow("depthright", 0);
    namedWindow("lastdisp",0);
    imshow("left", leftimg);
    imshow("right", rightimg);
    imshow("depthleft", depthleft);
    imshow("depthright", depthright);
    imshow("lastdisp",lastdisp);

    string strsave = "result_";
    imwrite(strsave +"depthleft.jpg", depthleft);
    imwrite(strsave +"depthright.jpg", depthright);
    imwrite(strsave +"lastdisp.jpg",lastdisp);
    waitKey(0);
    return 0;
}
