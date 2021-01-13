
#include"fenleiqi.h"
#include <algorithm>




void preProcessing(Mat &srcImg,Mat &binImg,int elementSize )
{






//方式2
	Mat grayImg;
	cvtColor(srcImg,grayImg,CV_RGB2GRAY);
	
	medianBlur(grayImg,grayImg,3);
	
	//blur(grayImg,grayImg, Size(3, 3));
	int blockSize = 25;  //块大小（奇数，大于1）
    int constValue = 35; //   差值（负值也可以）
    adaptiveThreshold(grayImg, binImg, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);// https://www.cnblogs.com/shangd/p/6094547.html
	//imshow("二值化",binImg);
	Mat element = getStructuringElement(MORPH_RECT, Size(elementSize, elementSize));
	//morphologyEx(binImg,binImg,MORPH_DILATE,element);
	dilate(binImg,binImg,element);
	//imshow("闭",binImg);
	medianBlur(binImg,binImg,5);
	//imshow("中值滤波",binImg);
	//blur(binImg, binImg, Size(3, 3));

	//OPENCV二值化图像内孔洞填充/小区域去除 https://blog.csdn.net/yansmile1/article/details/46761271
    Mat Dst = Mat::zeros(srcImg.size(), CV_8UC1);  
	 RemoveSmallRegion(binImg, binImg, 300, 1, 1);  //去除小区域
	 //imshow("局部_去除小区域",binImg);
   

}


void location(Mat &srcImg,Mat &binImg,double *sum_area,int *j,double *orignal_sum_area)
{
	//imshow("1",binImg);
    vector< vector<Point> > contours ;
	vector<Vec4i> hierarchy;
	double max_width=0;
	double sum_minAreaRect=0;
	double D;
	int m=0;
    if(binImg.data)
    {
        findContours(binImg,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	}


	//打印轮廓坐标点
	//for (int i=0;i<contours.size();i++)  
 //   {  
 //       for (int j=0;j<contours[i].size();j++)  
 //       {  
 //           cout<<contours[i][j].x<<"   "<<contours[i][j].y<<endl;  
 //        /*   ofstream f;  
 //           f.open("坐标轮廓线.txt",ios::out|ios::app);  
 //           f<<contours[i][j].x<<"  "<<contours[i][j].y<<endl;  */
 //       }  
 //   } 








	
// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	for (int i = 0; i < contours.size(); i++)
 {
 Moments moms = moments(Mat(contours[i]));
 double area = moms.m00;    //零阶矩即为二值图像的面积  double area = moms.m00;  
 //如果面积超出了设定的范围，则不再考虑该斑点  

 double area1 = contourArea(contours[i]);
 double Length=arcLength( contours[i],true );
 //绘制轮廓的最小外结矩形  
 RotatedRect rect=minAreaRect(contours[i]);  
   
 //一个循环，遍历所有部分，寻找最小面积的包围矩形
	for( unsigned int i = 0; i < contours.size(); i++ )
	{ 
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );//用指定精度逼近多边形曲线 
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );//计算点集的最外面（up-right）矩形边界
		//boundRect[i] = boundingRect( Mat(contours_poly[i]) );//计算点集的最外面（up-right）矩形边界
		//minEnclosingCircle( contours_poly[i], center[i], radius[i] );//对给定的 2D点集，寻找最小面积的包围圆形
		 RotatedRect rect=minAreaRect(contours[i]); 
	}
 

 if (area > 100 && area < 100000000 && Length>120)

 {
	 
	 //画出轮廓
 //drawContours(srcImg, contours, i, Scalar(0, 0, 255),1, 8, hierarchy, 0, Point());
 drawContours(srcImg, contours, i, Scalar(0, 0, 255),2, 8, hierarchy, 0, Point());
 // drawContours(srcImg, contours, i, Scalar(0, 0, 255),CV_FILLED, 8, hierarchy, 0, Point());
  

  *sum_area=*sum_area+contourArea(contours[i]);
  //drawContours(srcImg, contours, i, Scalar(0, 0, 255), 1, 8, hierarchy, 0, Point());
 //绘制minAreaRect
  vector<Point>min_rectangle;
     Point2f P[4];  
       rect.points(P);  
        for(int j=0;j<=3;j++)  
       {  
            line(srcImg,P[j],P[(j+1)%4],Scalar(255,0,0),1);
		   
        }

//打印每个轮廓minAreaRect的长和宽
 
//printf(" >通过minAreaRect计算出[%d]的长: Length = %.2f ,宽: Width = %.2f \n ",i,max(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height),min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height));
 
	 sum_minAreaRect=sum_minAreaRect + minAreaRect(contours[i]).size.width*minAreaRect(contours[i]).size.height;
//获取所有轮廓中最大的宽度的轮廓
 if(min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height)>max_width)
  {
 max_width=min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height);
  }
 
 //绘制在外接矩形
 //rectangle( srcImg, boundRect[i].tl(), boundRect[i].br(), Scalar(0,200,0), 2, 8, 0 );
 //printf(" >通过m00计算出轮廓[%d]的面积: (M_00) = %.2f \n OpenCV函数计算出的面积=%.2f, 长度: %.2f ,平均宽度: %.2f , \n\n\n", i, moms.m00, contourArea(contours[i]), arcLength( contours[i], true ) , contourArea(contours[i])/arcLength( contours[i], true ));
	//printf(" >通过m00计算出轮廓[%d]的面积: (M_00) = %.2f \n OpenCV函数计算出的面积=%.2f, 长度: %.2f ,平均宽度: %.2f , \n\n", i, moms.m00, contourArea(contours[i]), arcLength( contours[i], true ) , contourArea(contours[i])/boundRect[i].width);	
 


 *j = *j + 1;

 }
 else if (area >= 0 && area <= 10)
 {
 //drawContours(srcImg, contours, i, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy, 0, Point());

 m = m + 1;

 }


}

*orignal_sum_area=*sum_area;
*sum_area=*sum_area/float(srcImg.cols*srcImg.rows);//必须加float否则按int整除
 D=*sum_area/sum_minAreaRect;


}




//OPENCV二值化图像内孔洞填充/小区域去除
//https://blog.csdn.net/yansmile1/article/details/46761271
//https://blog.csdn.net/patkritlee/article/details/53380419
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)  
{     
    int RemoveCount=0;       //记录除去的个数  
    //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
    Mat Pointlabel = Mat::zeros( Src.size(), CV_8UC1 );  
      
    if(CheckMode==1)  
    {  
        //cout<<"Mode: 去除小区域. ";  
        for(int i = 0; i < Src.rows; ++i)    
        {    
            uchar* iData = Src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < Src.cols; ++j)    
            {    
                if (iData[j] < 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }    
    }  
    else  
    {  
        //cout<<"Mode: 去除孔洞. ";  
        for(int i = 0; i < Src.rows; ++i)    
        {    
            uchar* iData = Src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < Src.cols; ++j)    
            {    
                if (iData[j] > 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
       }    
    }  
  
    vector<Point2i> NeihborPos;  //记录邻域点位置  
    NeihborPos.push_back(Point2i(-1, 0));  
    NeihborPos.push_back(Point2i(1, 0));  
    NeihborPos.push_back(Point2i(0, -1));  
    NeihborPos.push_back(Point2i(0, 1));  
    if (NeihborMode==1)  
    {  
        //cout<<"Neighbor mode: 8邻域."<<endl;  
        NeihborPos.push_back(Point2i(-1, -1));  
        NeihborPos.push_back(Point2i(-1, 1));  
        NeihborPos.push_back(Point2i(1, -1));  
        NeihborPos.push_back(Point2i(1, 1));  
    }  
    else cout<<"Neighbor mode: 4邻域."<<endl;  
    int NeihborCount=4+4*NeihborMode;  
    int CurrX=0, CurrY=0;  
    //开始检测  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 0)    
            {    
                //********开始该点处的检查**********  
                vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
                GrowBuffer.push_back( Point2i(j, i) );  
                Pointlabel.at<uchar>(i, j)=1;  
                int CheckResult=0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  
  
                for ( int z=0; z<GrowBuffer.size(); z++ )  
                {  
  
                    for (int q=0; q<NeihborCount; q++)                                      //检查四个邻域点  
                   {  
                        CurrX=GrowBuffer.at(z).x+NeihborPos.at(q).x;  
                        CurrY=GrowBuffer.at(z).y+NeihborPos.at(q).y;  
                        if (CurrX>=0&&CurrX<Src.cols&&CurrY>=0&&CurrY<Src.rows)  //防止越界  
                        {  
                            if ( Pointlabel.at<uchar>(CurrY, CurrX)==0 )  
                            {  
                                GrowBuffer.push_back( Point2i(CurrX, CurrY) );  //邻域点加入buffer  
                                Pointlabel.at<uchar>(CurrY, CurrX)=1;           //更新邻域点的检查标签，避免重复检查  
                            }  
                        }  
                    }  
  
                }  
                if (GrowBuffer.size()>AreaLimit) CheckResult=2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
                else {CheckResult=1;   RemoveCount++;}  
                for (int z=0; z<GrowBuffer.size(); z++)                         //更新Label记录  
                {  
                    CurrX=GrowBuffer.at(z).x;   
                    CurrY=GrowBuffer.at(z).y;  
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;  
                }  
                //********结束该点处的检查**********  
  
  
            }    
        }    
    }    
  
    CheckMode=255*(1-CheckMode);  
    //开始反转面积过小的区域  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iData = Src.ptr<uchar>(i);  
        uchar* iDstData = Dst.ptr<uchar>(i);  
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 2)    
            {    
                iDstData[j] = CheckMode;   
            }    
            else if(iLabel[j] == 3)  
            {  
                iDstData[j] = iData[j];  
            }  
        }    
    }   
      
    //cout<<RemoveCount<<" objects removed."<<endl;

} 

void touying(Mat &src,float *Y_diff,float *X_diff)

{
	Mat src_gray,src_binary,paintX,paintY;

	//创建两个图像框，用于绘制投影图  (黑底，0 黑，  1 白)
    paintX = Mat::zeros( src.rows, src.cols, CV_8UC1 );       
	paintY = Mat::zeros( src.rows, src.cols, CV_8UC1 );

	//Mat paintX( src.cols, src.rows, CV_8UC1, Scalar( 0, 0, 0));
	//Mat paintY( src.cols, src.rows, CV_8UC1, Scalar( 0, 0, 0));
	//转化为灰度图像
	//cout<<"paintX.cols = "<<paintX.cols<<endl;
	//cout<<"paintX.rows = "<<paintX.rows<<endl;
	 
	
	//二值化
	//方式1
	preProcessing(src,src_binary,5);
	//方式2
	/* Mat grayImg;
	cvtColor(src,grayImg,CV_RGB2GRAY);
	medianBlur(grayImg,grayImg,5);
	int blockSize = 25;  
    int constValue = 35;    
    adaptiveThreshold(grayImg, src_binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);*/
  
	

	//OPENCV二值化图像内孔洞填充/小区域去除 https://blog.csdn.net/yansmile1/article/details/46761271
	  double t = (double)getTickCount();  
	 char* OutPath = "局部_去除孔洞.jpg";  
    Mat Dst = Mat::zeros(src.size(), CV_8UC1);  
	 RemoveSmallRegion(src_binary, src_binary, 300, 1, 1);  //去除小区域
    //RemoveSmallRegion(src_binary, src_binary, 3000, 0, 0);  //填充空洞，所以不采用
	  //imshow("Dst",src_binary);
    /*cout<<"Done!"<<endl; 
	imshow("Dst",Dst);
    imwrite(OutPath, Dst);        
    t = ((double)getTickCount() - t)/getTickFrequency();  
    cout<<"Time cost: "<<t<<" sec."<<endl; */





	int* v = new int[src.cols*2];
	int* h = new int[src.rows*2];
	//cout<<"src.cols = "<<src.cols<<endl;
	//cout<<"src.rows = "<<src.rows<<endl;
	memset(v, 0, src.cols*4);
	memset(h, 0, src.rows*4);
         //方法一的实现
	
	int i,j;
	float X_max=0,X_min=721, Y_max=0,Y_min=721;
	float X=0,Y=0;
   //垂直方向进行累加（积分）
	for( i=0; i<src_binary.cols; i++)           //列
	{
		for( j=0; j<src_binary.rows; j++)                //行
		{
			if( src_binary.at<uchar>( j, i ) == 255)                //统计的是白色像素的数量
				v[i]++;
		}
		
		

		if(v[i]>X_max)
		{
			X_max=v[i];
		}

		if(v[i]<X_min)
		{
			X_min=v[i];
		}
	}
	// *X_diff= (X_max-X_min);
	//printf("X_max=%d X_min=%d X_diff=%d \n",X_max,X_min,X_diff);
	//绘制垂直方向上的投影
	for( i=0; i<src_binary.cols; i++)
	{	
		for( j=0; j<v[i]; j++)
		{
			paintX.at<uchar>( j, i ) = 255;        //填充白色的像素
			X=X+v[i];
		}
	}
	*X_diff= (X_max-X_min)/ X;



	//水平方向进行累加（积分）
	for( i=0; i<src_binary.rows; i++)           //行
	{
		for( j=0; j<src_binary.cols; j++)                //列
		{
			if( src_binary.at<uchar>( i, j ) == 255)       //统计白色像素的数量
				h[i]++;
		}

		

		if(h[i]>Y_max)
		{
			Y_max=h[i];
		}

		if(h[i]<Y_min)
		{
			Y_min=h[i];
		}
	}
	// *Y_diff=(Y_max-Y_min);
	//printf("Y_max=%d Y_min=%d Y_diff=%d \n",Y_max,Y_min,Y_diff);
	//绘制水平方向上的投影
	for( i=0; i<src_binary.rows; i++)
	{	
		for( j=0; j<h[i]; j++)
		{
			paintY.at<uchar>( i, j ) = 255;        //填充色的像素
			Y=Y+h[i];
		}
	}
	 *Y_diff=(Y_max-Y_min)/Y;
      //printf("*X_diff=%f *Y_diff=%f \n",*X_diff,*Y_diff); 
	
	//显示图像
	//imshow("wnd_binary", src_binary);
	//imshow("wnd_X", paintX);
	//imshow("wnd_Y", paintY);
	
	


}

//------------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------三种裂缝宽度计算方式（纵向横向，斜向）---------------------------------------------------------------//

void measure_v(Mat &srcImg, double *maxdiff_width,double *mindiff_width)
{

	//...求最大宽度（根据左右边界红色像素的坐标求最大距离,分为两侧，只扫描同一行的左右两个坐标求距离）...//
	vector<Point> p0,p1,p2;

	
	
	int i,j;
	int k=1;
    int cPointR1,cPointG1,cPointB1,cPointR2,cPointG2,cPointB2,cPointR3,cPointG3,cPointB3;//currentPoint;
	
	int t=0;
	
	int width;
	
    for(i=0;i<srcImg.rows;i++)
	{   
		
		int s[720]={0};
	    int max_j=s[0],min_j=srcImg.cols;
        for(j=0;j<srcImg.cols;j++)
        {
			
			 
            cPointB1=srcImg.at<Vec3b>(i,j)[0];
            cPointG1=srcImg.at<Vec3b>(i,j)[1];
            cPointR1=srcImg.at<Vec3b>(i,j)[2];



			if(cPointB1==0&cPointG1==0&cPointR1==255)
                {
					s[j]=j;//此处并没有把点集存储之后再取最大最小值，因为会出现多线程问题
					
					//if(k%2!=0)
					//{
					//	//printf("第一组(%d,%d)\n",j,i);//没有另起一行应为打印数据太多，窗口只能显示一部分；vector(y,x)存储，所以此处（j,i）打印
					//    p1.push_back(Point(j,i));
					//}
					//else if(k<=2) //**此处不可为else if,参考if嵌套语句
					//{
					//	//printf("第二组(%d,%d)\n",j,i); 
					//    p2.push_back(Point(j,i));
					//}

					//k=k+1;
                } 
			else
				s[j]=0;
			


			if(s[j]>max_j)
			{
				max_j=s[j];
			}
			if ((0<s[j])&&(s[j]<min_j))
			{
				min_j=s[j];
			}

			s[j]=0;

          }
		//printf("min_j=%d",min_j);
		//printf("max_j=%d  ",max_j);

		if((min_j!=srcImg.cols)&&(max_j!=0))
		{
		
		//printf("第一组(%d,%d)",min_j,i);//没有另起一行应为打印数据太多，窗口只能显示一部分；vector(y,x)存储，所以此处（j,i）打印
		
					    p1.push_back(Point(min_j,i));
		}

		
		
		if((min_j!=srcImg.cols)&&(max_j!=0))
		
		{
		//printf("第二组(%d,%d) ",max_j,i); 
					    p2.push_back(Point(max_j,i));
		}
		if((min_j!=srcImg.cols)&&(max_j!=0))
		{
			width=max_j-min_j;
		}


		 if(width>*maxdiff_width)
		 {
			 *maxdiff_width=width;

		 }

		if((width>0)&&(width<*mindiff_width))
		 {
			 *mindiff_width=width;

		 }		
		
	}
	
	//printf("\n mindiff_width=%f,maxdiff_width=%f \n ",*mindiff_width,*maxdiff_width); 




//--------------------------------------求最大宽度的最佳方法（裂缝分为两侧，每一侧每一点寻找另外一侧的最短距离），理论上最好，实际不怎么行-------------------------------------------------------------//
	//如果轮廓填充为外边界，则画出的最小外接矩形框会影响宽度（蓝色边框有和轮廓边界重合的地方），如果轮廓填充为实则不影响
	
    //求最大宽度。http://bbs.csdn.net/topics/391056710
	float p=0;
	float min_p=720;//图像宽高中最大值
	vector<float> max_p;
	int q=0;
	int v=0;


	//方式1 全部遍历
	//for (vector<Point> ::iterator iter1 = p1.begin(); iter1 != p1.end(); ++iter1) 

	//{
	//
	//
	//	
	//	for (vector<Point> ::iterator iter2 = p2.begin(); iter2 != p2.end();++iter2)
	//	
	//	{

	//		 p=sqrt(float(((*iter1).x-(*iter2).x)*((*iter1).x-(*iter2).x)+((*iter1).y-(*iter2).y)*((*iter1).y-(*iter2).y)));

	//		 //p=abs(float(abs((*iter1).x-(*iter2).x)+abs((*iter1).y-(*iter2).y)));
	//		 //p=(abs((*iter1).x - (*iter2).x) + abs((*iter1).y - (*iter2).y));
	//		 
	//		 if(((6<p)&&p<min_p))//10<p从打印出数据得出得避免邻近一点变多点数据误差，其原因是画轮廓时锯齿形线造成
	//			//if(p<min_p)

	//		 {
	//	      min_p=p;
	//        }
	//	}
	//	max_p.push_back(min_p);
 //     	
	//}

	//方式2 附近10个点

	for (vector<Point> ::iterator iter1 = p1.begin(); iter1 != p1.end(); ++iter1) 

	{
	
	
		q++;
		v++;
		//for (vector<Point> ::iterator iter2 = p2.begin(); iter2 != p2.end();++iter2)
		if((10<q)&&((v+10)<p1.size()))
		{

		
		for (vector<Point> ::iterator iter2 = p2.begin()+(q-10); iter2 != p2.begin()+(v+10);++iter2) 
		
		{

			 p=sqrt(float(((*iter1).x-(*iter2).x)*((*iter1).x-(*iter2).x)+((*iter1).y-(*iter2).y)*((*iter1).y-(*iter2).y)));

			 //p=abs(float(abs((*iter1).x-(*iter2).x)+abs((*iter1).y-(*iter2).y)));
			 //p=(abs((*iter1).x - (*iter2).x) + abs((*iter1).y - (*iter2).y));
			 
			 //if(((6<p)&&p<min_p))//10<p从打印出数据得出得避免邻近一点变多点数据误差，其原因是画轮廓时锯齿形线造成
				if(p<min_p)

			 {
		      min_p=p;
	        }
		}
		max_p.push_back(min_p);
      }

		
	}



	

	float Max_Width = *max_element(max_p.begin(),max_p.end());//https://www.cnblogs.com/sword-/p/8036813.html
	float Min_Width = *min_element(max_p.begin(),max_p.end());
	*maxdiff_width=Max_Width;
	//printf("Min_Width=%f,Max_Width=%f \n",Min_Width,Max_Width);
//---------------------------------------------------------------------------------------------------------------------//
	 

			   
}

void measure_l(Mat &srcImg, double *maxdiff_width,double *mindiff_width)
{

	//...求最大宽度（根据左右边界红色像素的坐标求最大距离）...//
	vector<Point> p0,p1,p2;

	
	
	int i,j;
	int k=1;
    int cPointR1,cPointG1,cPointB1,cPointR2,cPointG2,cPointB2,cPointR3,cPointG3,cPointB3;//currentPoint;
	
	int t=0;
	
	int width;
	
    for(i=0;i<srcImg.cols;i++)
	{   
		
		int s[480]={0};
	    int max_j=s[0],min_j=srcImg.rows;
        for(j=0;j<srcImg.rows;j++)
        {
			
			 
            cPointB1=srcImg.at<Vec3b>(j,i)[0];//**此处特别注意必须为（rows，cols）
            cPointG1=srcImg.at<Vec3b>(j,i)[1];
            cPointR1=srcImg.at<Vec3b>(j,i)[2];
			if(cPointB1==0&cPointG1==0&cPointR1==255)
                {
					s[j]=j;//此处并没有把点集存储之后再取最大最小值，因为会出现多线程问题
					
                } 
			else
				s[j]=0;
			

			if(s[j]>max_j)
			{
				max_j=s[j];
			}
			if ((0<s[j])&&(s[j]<min_j))
			{
				min_j=s[j];
			}
			s[j]=0;


          }
		//printf("min_j=%d",min_j);
		//printf("max_j=%d  ",max_j);

		if((min_j!=srcImg.rows)&&(max_j!=0))
		{
		
		//printf("第一组(%d,%d)",min_j,i);//没有另起一行应为打印数据太多，窗口只能显示一部分；vector(y,x)存储，所以此处（j,i）打印
		
					    p1.push_back(Point(min_j,i));
		}

		
		
		if((min_j!=srcImg.rows)&&(max_j!=0))
		
		{
		//printf("第二组(%d,%d ) ",max_j,i); 
					    p2.push_back(Point(max_j,i));
		}
		if((min_j!=srcImg.rows)&&(max_j!=0))
		{
			width=max_j-min_j;
		}


		 if(width>*maxdiff_width)
		 {
			 *maxdiff_width=width;

		 }

		if((width>0)&&(width<*mindiff_width))
		 {
			 *mindiff_width=width;

		 }		
		
	}
	//printf("\n mindiff_width=%f,maxdiff_width=%f \n ",*mindiff_width,*maxdiff_width); 




//--------------------------------------求最大宽度的最佳方法（裂缝分为两侧，每一侧每一点寻找另外一侧的最短距离），理论上最好，实际不怎么行-------------------------------------------------------------//
	
////求最大宽度。http://bbs.csdn.net/topics/391056710
//如果轮廓填充为外边界，则画出的最小外接矩形框会影响宽度（蓝色边框有和轮廓边界重合的地方），如果轮廓填充为实则不影响
	float p=0;
	float min_p=480;//图像宽高中最大值
	vector<float> max_p;
	
	int q=0;
	int v=0;


	//方式1 全部遍历
	//for (vector<Point> ::iterator iter1 = p1.begin(); iter1 != p1.end(); ++iter1) 

	//{
	//
	//
	//	
	//	for (vector<Point> ::iterator iter2 = p2.begin(); iter2 != p2.end();++iter2)
	//	
	//	{

	//		 p=sqrt(float(((*iter1).x-(*iter2).x)*((*iter1).x-(*iter2).x)+((*iter1).y-(*iter2).y)*((*iter1).y-(*iter2).y)));

	//		 //p=abs(float(abs((*iter1).x-(*iter2).x)+abs((*iter1).y-(*iter2).y)));
	//		 //p=(abs((*iter1).x - (*iter2).x) + abs((*iter1).y - (*iter2).y));
	//		 
	//		 if(((6<p)&&p<min_p))//10<p从打印出数据得出得避免邻近一点变多点数据误差，其原因是画轮廓时锯齿形线造成
	//			//if(p<min_p)

	//		 {
	//	      min_p=p;
	//        }
	//	}
	//	max_p.push_back(min_p);
 //     	
	//}


	//方式2 附近10个点
	for (vector<Point> ::iterator iter1 = p1.begin(); iter1 != p1.end(); ++iter1) 

	{
	
	
		q++;
		v++;
		//for (vector<Point> ::iterator iter2 = p2.begin(); iter2 != p2.end();++iter2)
		if((10<q)&&((v+10)<p1.size()))
		{

		
		for (vector<Point> ::iterator iter2 = p2.begin()+(q-10); iter2 != p2.begin()+(v+10);++iter2) 
		
		{

			 p=sqrt(float(((*iter1).x-(*iter2).x)*((*iter1).x-(*iter2).x)+((*iter1).y-(*iter2).y)*((*iter1).y-(*iter2).y)));

			 //p=abs(float(abs((*iter1).x-(*iter2).x)+abs((*iter1).y-(*iter2).y)));
			 //p=(abs((*iter1).x - (*iter2).x) + abs((*iter1).y - (*iter2).y));
			 
			 //if(((6<p)&&p<min_p))//10<p从打印出数据得出得避免邻近一点变多点数据误差，其原因是画轮廓时锯齿形线造成
				if(p<min_p)

			 {
		      min_p=p;
	        }
		}
		max_p.push_back(min_p);
      }

		
	}
//---------------------------------------------------------------------------------------------------------------------------//
	float Max_Width = *max_element(max_p.begin(),max_p.end());//https://www.cnblogs.com/sword-/p/8036813.html
	float Min_Width = *min_element(max_p.begin(),max_p.end());
	*maxdiff_width=Max_Width;
	//printf("Min_Width=%f,Max_Width=%f \n",Min_Width,Max_Width);
			   
}

void measure_s(Mat &srcImg, double *maxdiff_width,double *mindiff_width)
{

	double v_maxdiff_width=0;
	double v_mindiff_width=max(srcImg.rows,srcImg.cols);
	double l_maxdiff_width=0;
	double l_mindiff_width=max(srcImg.rows,srcImg.cols);

  measure_v(srcImg,&v_maxdiff_width,&v_mindiff_width);
  measure_l(srcImg,&l_maxdiff_width,&l_mindiff_width);
  *maxdiff_width=0.5*min(v_maxdiff_width,l_maxdiff_width);
  *mindiff_width=min(v_mindiff_width,l_mindiff_width);


}
//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//







