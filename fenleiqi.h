#include "opencv2/core/core.hpp"
#include "opencv2//highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <time.h>
#include <stdlib.h>
#include <opencv2\opencv.hpp>
#include <vector>
#include<time.h>  
#include "stdio.h"
#include "cv.h" 
#include "highgui.h"  
#include <cxcore.hpp>
#include <afxcom_.h>
#include<math.h>
#include <iostream>       
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
using namespace std;
using namespace cv;
#include <algorithm>

void preProcessing(Mat &srcImg,Mat &binImg,int elementSize);
void location(Mat &srcImg,Mat &binImg,double *sum_area,int *j,double *orignal_sum_area);
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);
//void touying(Mat &src,int *Y_diff,int *X_diff);
void touying(Mat &src,float *Y_diff,float *X_diff);
void measure_v(Mat &srcImg, double *maxdiff_width,double *mindiff_width);
void measure_l(Mat &srcImg, double *maxdiff_width,double *mindiff_width);
void measure_s(Mat &srcImg, double *maxdiff_width,double *mindiff_width);