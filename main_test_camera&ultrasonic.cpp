// cracker_find.cpp : �������̨Ӧ�ó������ڵ㡣
//

//#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <vector>
#include<time.h> 
#include <iostream> 
#include "stdio.h"
#include "cv.h" 
#include "highgui.h"  
#include <cxcore.hpp>
#include <afxcom_.h>
#include<stdio.h>
#include<math.h>
#include <iostream>     
#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include<algorithm> //����vectorxѰ�����ֵ�õ�




//#include "stdafx.h"

#include <fstream> 
#include <strstream>
#include "opencv2/opencv.hpp" 
#include <io.h>
#include <vector> 
#include "vector"
#include <stdio.h>
#include <tchar.h>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "time.h"
#include "SerialPort.h"


using namespace cv;
using namespace std;

#include"fenleiqi.h"




class NumTrainData  
{  
public:  
	NumTrainData()  
	{  
		memset(data, 0, sizeof(data));  
		result = -1;  
	}  
public:  
	float data[512];  
	int result;  
};  

vector<NumTrainData> buffer;  
int featureLen = 4;  


char * filePath = "..\\train_two"; 
typedef struct PathFile
{
	string fileName;
	string files;
};
void getFiles( string path, vector<PathFile>& files )  
{  
	//�ļ����  
	long   hFile   =   0;  
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;  
	string p; 
	PathFile temp={};
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
	{  
		do  
		{  
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if((fileinfo.attrib &  _A_SUBDIR))  
			{  
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
					getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
			}  
			else  
			{  
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
				temp.fileName=fileinfo.name;
				temp.files=p.assign(path).append("\\").append(fileinfo.name);
				files.push_back(temp);
			}  
		}while(_findnext(hFile, &fileinfo)  == 0);  
		_findclose(hFile);  
	}  
}  

int ReadTrainData()  
{  

	//Create source and show image matrix 
	vector<PathFile> files;  

	////��ȡ��·���µ������ļ�  
	getFiles(filePath, files );
	//Mat src = Mat::zeros(rows, cols, CV_8UC1);  
	//Mat temp = Mat::zeros(8, 8, CV_8UC1);  
	Mat img, dst;  

	char label = 0;  
	Scalar templateColor(255, 0, 255 );  

	NumTrainData rtd;  
 
	int total = 0;  

	for(;total<files.size();total++)  
	{  
		string nstr=files[total].fileName.substr(0,1);//ͼƬ����ĸ��Ϊlabel
		rtd.result = nstr[0];//��ǰ�ַ���ASCII��
		Mat img=imread(files[total].files);
		resize(img, img, Size(360,240));
		//memcpy(rtd.data,img.data,img.cols*img.rows);
		//resize(img, temp, temp.size());  
		//threshold(img, img, 100, 255, CV_THRESH_BINARY);//��ֵ��



	int height=img.rows;  
    int width=img.cols;
	//int X_diff=0,Y_diff=0;
	float X_diff=0,Y_diff=0;
	double sum_area=0,orignal_sum_area;
	double maxdiff_width=0;
	double mindiff_width=max(img.rows,img.cols);
	int j=0;
	float v[4];
	Mat m = Mat::zeros(1, featureLen, CV_32FC1);
	//printf("ͼ���=%d,��=%d \n",width,height);
	//imshow("srcImg",img);	
	Mat binImg;
	
	preProcessing(img,binImg,3);
	 
	touying(img,&Y_diff,&X_diff);//�������ض��ֵhttp://blog.csdn.net/qq_23100787/article/details/50288217
	 //printf("X_diff=%d,Y_diff=%d \n",X_diff,Y_diff);
	 v[0]=X_diff;
	 v[1]=Y_diff;

	location(img,binImg,&sum_area,&j,&orignal_sum_area);
	//printf("sum_Area=%f \n",sum_area);
	//printf("j= %d \n", j);

	v[2]=sum_area;
	//v[3]=j;
	
	v[3]=1;
	
	/*v[0]=sum_area;
	v[1]=j;
	v[2]=X_diff;
	v[3]=Y_diff;
*/

	 if(j<=0)
 {

 string txt3 = "no crack " ;
 putText(img, txt3, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
 }
 else
     {
 

		for(int i = 0; i<4; i++) //����άͼ��תΪһά���� 
		{  
			
				rtd.data[i] = v[i];  
			 
		}  
		buffer.push_back(rtd);//����ȫ��ģ��ͼ��һά���鵽buffer��  
		cout<<rtd.result<<"  ";
		if(rtd.result=='1')
		{
			measure_v(img,&maxdiff_width,&mindiff_width);

			string txt4 = "lengthways crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxdiff_width : " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
 //imshow("ansImg",img);
       // waitKey(1);



		}
		else if(rtd.result=='2')
		{
			measure_l(img,&maxdiff_width,&mindiff_width);

			string txt4 = "transverse crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxdiff_width : " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
 //imshow("ansImg",img);
       // waitKey(1);
		
		}
		else if(rtd.result=='3')
		{
            measure_s(img,&maxdiff_width,&mindiff_width);

			string txt4 = "slant crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxdiff_width : " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
	//imshow("ansImg",img);
        //waitKey(1);	
		
		
		
		}
		else if(rtd.result=='4')
		{
			measure_s(img,&maxdiff_width,&mindiff_width);

			string txt4 = "net crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxdiff_width : " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
	         }

	  }

imshow("ansImg",img);
        waitKey(30);

  }  

	return total;  
}  

void newSvmStudy(vector<NumTrainData>& trainData)  
{  
	int testCount = trainData.size();  

	Mat m = Mat::zeros(1, featureLen, CV_32FC1); //1��512�о��� 512=32*16 ��һ��ͼ�����ݴ�С
	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);//�������С�512�о���  
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);  

	for (int i= 0; i< testCount; i++)   
	{   

		NumTrainData td = trainData.at(i);  
		memcpy(m.data, td.data, featureLen*sizeof(float));  
		normalize(m, m);  
		memcpy(data.data + i*featureLen*sizeof(float), m.data, featureLen*sizeof(float));  

		res.at<unsigned int>(i, 0) = td.result; //�洢�ַ�ASCII�� 
	}  

	/////////////START SVM TRAINNING//////////////////

	////RBF
	CvSVM svm /*= CvSVM()*/;   
	CvSVMParams param;   
	CvTermCriteria criteria;  
	criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);   
	param= CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);   

	svm.train(data, res, Mat(), Mat(), param);  
	svm.save( "SVM_DATA.xml" );

	//����
	//CvSVM svm /*= CvSVM()*/;   
	 /* CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	svm.train(data, res, Mat(), Mat(), param);  
	svm.save( "SVM_DATA.xml" );
	*/
	
	
	//���Բ��ɷ�
	//CvSVM svm /*= CvSVM()*/;   
	//CvSVMParams params;
 //   params.svm_type    = SVM::C_SVC;
 //   params.C           = 0.1;
 //   params.kernel_type = SVM::LINEAR;
 //   params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
	//svm.train(data, res, Mat(), Mat(), params);  
	//svm.save( "SVM_DATA.xml" );



} 


 char * filePath1 = "..\\test_two"; 
 int newSvmPredict()  
{  


////----- ��Ƶ����-----//
////--------------------------//

	CvSVM svm ;   
	svm.load( "SVM_DATA.xml" ); 
	vector<PathFile> files;
	bool stop = false;
	Mat img;
	//-------------����ͷ--------------------------------//
	VideoCapture cap(-1);    //��Ĭ������ͷ
	if (!cap.isOpened())
	{
		return -1;
	}
	//-------------��Ƶ------------------------------------//
//VideoCapture cap("C:\\Users\\gxm\\Desktop\\720 480���ͼ���ͳ�����5.16��������ʹ�� _720 480ͼƬѵ������5.8��\\test2�޸İ�1\\test2\\2.avi");    //����Ƶ
	

// ��ȡ��Ƶ��֡��///
	long totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);///
	//cout << "total frames: " << totalFrameNumber << endl;///
	long currentFrame = 0;///




float Distance;


	while (!stop)
	{
		cap >> img;
		char path[255];
		Mat dst;
		resize(img, img, Size(360,240)); 
		Mat m = Mat::zeros(1, featureLen, CV_32FC1); 
		

		int height=img.rows;  
    int width=img.cols;
	float X_diff=0,Y_diff=0;
	double sum_area=0,orignal_sum_area=0;
	double maxdiff_width=0;
	double mindiff_width=max(img.rows,img.cols);
	int j=0;
	float O[4];

	// ����ÿ30֡��ȡһ��֡///
	if (currentFrame % 5 == 0)///
{///

	//printf("ͼ���=%d,��=%d \n",width,height);
	imshow("srcImg",img);	
	Mat binImg;
	
	preProcessing(img,binImg,3);
	 
	touying(img,&Y_diff,&X_diff);//�������ض��ֵhttp://blog.csdn.net/qq_23100787/article/details/50288217
	 //printf("X_diff=%d,Y_diff=%d \n",X_diff,Y_diff);
	 O[0]=X_diff;
	 O[1]=Y_diff;

	location(img,binImg,&sum_area,&j,&orignal_sum_area);
	//printf("sum_Area=%f \n",sum_area);
	//printf("j= %d \n", j);

	O[2]=sum_area;
	//O[3]=j;
	O[3]=1;
	
	//��ʾ����������
	receive_distance( &Distance);
	if(Distance==0)
	{
		Distance=10;
	}
	printf("Distance=%.2lf",Distance);
	printf("%s\n","cm");
	

	if(j<=0)
 {

 string txt3 = "no crack " ;
 putText(img, txt3, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
 }
 else
{

   for(int i = 0; i<4; i++) //����άͼ��תΪһά���� 
		{  
			 
			  
				m.at<float>(i) = O[i]; 
			  
		}  
		normalize(m, m);  
		char ret = (char)svm.predict(m);   
		//cout<<ret<<"  ";
		if(ret=='1')
		{
			measure_v(img,&maxdiff_width,&mindiff_width);

			string txt4 = "lengthways crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxwidth: " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);

		}
		else if(ret=='2')
		{
			measure_l(img,&maxdiff_width,&mindiff_width);

			string txt4 = "transverse crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxwidth: " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);

		}
		else if(ret=='3')
		{
            measure_s(img,&maxdiff_width,&mindiff_width);

			string txt4 = "slant crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxwidth: " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
	
		}
		else if(ret=='4')
		{
			//measure_v(img,&maxdiff_width,&mindiff_width);
			measure_s(img,&maxdiff_width,&mindiff_width);
			string txt4 = "net crack " ;
            putText(img, txt4, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
            Scalar(255, 0, 0), 2, 8);

			char t[256];
 sprintf_s(t, "%.2f", maxdiff_width);
 string s = t;
 string txt = "maxwidth: " + s;
 putText(img, txt, Point(20, 120), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char w[256];
 sprintf_s(w, "%01d", j);
 string e = w;
 string txt1 = "crack count : " + e;
 putText(img, txt1, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(100, 0, 10), 2, 8);

 char u[256];
 sprintf_s(u, "%.2f", orignal_sum_area);
 string g = u;
 string txt2 = "SumArea : " + g;
 putText(img, txt2, Point(20, 90), CV_FONT_HERSHEY_COMPLEX, 1,
 Scalar(255, 0, 0), 2, 8);
	         }

		
       // waitKey(100);

		}
   imshow("ansImg",img);
  //if (waitKey(300) >= 0)//���ص�����10ms�������һ��Keyboardֵ���������ֵһ����ֵ���򷵻ؼ�ֵ�����򷵻�-1.
  //  stop = true;

   img.release();
   char c=1;
    c = cvWaitKey(1);
	if (c == 32) break;


 }///
// ��������///
		if (currentFrame >= totalFrameNumber)///
		{
			//stop = false;///
		}
		currentFrame++;///

 }

    getchar();
	return 0; 
  }

//-------------------------------------------//
//----------------------------------------------//



int main(int argc, char** argv[])
{ 
	//ѵ��������,��������ǰ����
	//int nowCount=ReadTrainData();  
   
	//newSvmStudy(buffer); 



	CSerialPort mySerialPort;  
 
    if (!mySerialPort.InitPort(5))  
    {  
        std::cout << "initPort fail !" << std::endl;  
    }  
    else 
    {  
        std::cout << "initPort success !" << std::endl;  
    }  
 
    if (!mySerialPort.OpenListenThread())  
    {  
        std::cout << "OpenListenThread fail !" << std::endl;  
    }  
    else 
    {  
        std::cout << "OpenListenThread success !" << std::endl;  
    } 




   
	newSvmPredict();  

	return 0;
}




