
#include"fenleiqi.h"
#include <algorithm>




void preProcessing(Mat &srcImg,Mat &binImg,int elementSize )
{






//��ʽ2
	Mat grayImg;
	cvtColor(srcImg,grayImg,CV_RGB2GRAY);
	
	medianBlur(grayImg,grayImg,3);
	
	//blur(grayImg,grayImg, Size(3, 3));
	int blockSize = 25;  //���С������������1��
    int constValue = 35; //   ��ֵ����ֵҲ���ԣ�
    adaptiveThreshold(grayImg, binImg, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);// https://www.cnblogs.com/shangd/p/6094547.html
	//imshow("��ֵ��",binImg);
	Mat element = getStructuringElement(MORPH_RECT, Size(elementSize, elementSize));
	//morphologyEx(binImg,binImg,MORPH_DILATE,element);
	dilate(binImg,binImg,element);
	//imshow("��",binImg);
	medianBlur(binImg,binImg,5);
	//imshow("��ֵ�˲�",binImg);
	//blur(binImg, binImg, Size(3, 3));

	//OPENCV��ֵ��ͼ���ڿ׶����/С����ȥ�� https://blog.csdn.net/yansmile1/article/details/46761271
    Mat Dst = Mat::zeros(srcImg.size(), CV_8UC1);  
	 RemoveSmallRegion(binImg, binImg, 300, 1, 1);  //ȥ��С����
	 //imshow("�ֲ�_ȥ��С����",binImg);
   

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


	//��ӡ���������
	//for (int i=0;i<contours.size();i++)  
 //   {  
 //       for (int j=0;j<contours[i].size();j++)  
 //       {  
 //           cout<<contours[i][j].x<<"   "<<contours[i][j].y<<endl;  
 //        /*   ofstream f;  
 //           f.open("����������.txt",ios::out|ios::app);  
 //           f<<contours[i][j].x<<"  "<<contours[i][j].y<<endl;  */
 //       }  
 //   } 








	
// ����αƽ����� + ��ȡ���κ�Բ�α߽��
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	for (int i = 0; i < contours.size(); i++)
 {
 Moments moms = moments(Mat(contours[i]));
 double area = moms.m00;    //��׾ؼ�Ϊ��ֵͼ������  double area = moms.m00;  
 //�������������趨�ķ�Χ�����ٿ��Ǹðߵ�  

 double area1 = contourArea(contours[i]);
 double Length=arcLength( contours[i],true );
 //������������С������  
 RotatedRect rect=minAreaRect(contours[i]);  
   
 //һ��ѭ�����������в��֣�Ѱ����С����İ�Χ����
	for( unsigned int i = 0; i < contours.size(); i++ )
	{ 
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );//��ָ�����ȱƽ���������� 
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );//����㼯�������棨up-right�����α߽�
		//boundRect[i] = boundingRect( Mat(contours_poly[i]) );//����㼯�������棨up-right�����α߽�
		//minEnclosingCircle( contours_poly[i], center[i], radius[i] );//�Ը����� 2D�㼯��Ѱ����С����İ�ΧԲ��
		 RotatedRect rect=minAreaRect(contours[i]); 
	}
 

 if (area > 100 && area < 100000000 && Length>120)

 {
	 
	 //��������
 //drawContours(srcImg, contours, i, Scalar(0, 0, 255),1, 8, hierarchy, 0, Point());
 drawContours(srcImg, contours, i, Scalar(0, 0, 255),2, 8, hierarchy, 0, Point());
 // drawContours(srcImg, contours, i, Scalar(0, 0, 255),CV_FILLED, 8, hierarchy, 0, Point());
  

  *sum_area=*sum_area+contourArea(contours[i]);
  //drawContours(srcImg, contours, i, Scalar(0, 0, 255), 1, 8, hierarchy, 0, Point());
 //����minAreaRect
  vector<Point>min_rectangle;
     Point2f P[4];  
       rect.points(P);  
        for(int j=0;j<=3;j++)  
       {  
            line(srcImg,P[j],P[(j+1)%4],Scalar(255,0,0),1);
		   
        }

//��ӡÿ������minAreaRect�ĳ��Ϳ�
 
//printf(" >ͨ��minAreaRect�����[%d]�ĳ�: Length = %.2f ,��: Width = %.2f \n ",i,max(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height),min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height));
 
	 sum_minAreaRect=sum_minAreaRect + minAreaRect(contours[i]).size.width*minAreaRect(contours[i]).size.height;
//��ȡ�������������Ŀ�ȵ�����
 if(min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height)>max_width)
  {
 max_width=min(minAreaRect(contours[i]).size.width,minAreaRect(contours[i]).size.height);
  }
 
 //��������Ӿ���
 //rectangle( srcImg, boundRect[i].tl(), boundRect[i].br(), Scalar(0,200,0), 2, 8, 0 );
 //printf(" >ͨ��m00���������[%d]�����: (M_00) = %.2f \n OpenCV��������������=%.2f, ����: %.2f ,ƽ�����: %.2f , \n\n\n", i, moms.m00, contourArea(contours[i]), arcLength( contours[i], true ) , contourArea(contours[i])/arcLength( contours[i], true ));
	//printf(" >ͨ��m00���������[%d]�����: (M_00) = %.2f \n OpenCV��������������=%.2f, ����: %.2f ,ƽ�����: %.2f , \n\n", i, moms.m00, contourArea(contours[i]), arcLength( contours[i], true ) , contourArea(contours[i])/boundRect[i].width);	
 


 *j = *j + 1;

 }
 else if (area >= 0 && area <= 10)
 {
 //drawContours(srcImg, contours, i, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy, 0, Point());

 m = m + 1;

 }


}

*orignal_sum_area=*sum_area;
*sum_area=*sum_area/float(srcImg.cols*srcImg.rows);//�����float����int����
 D=*sum_area/sum_minAreaRect;


}




//OPENCV��ֵ��ͼ���ڿ׶����/С����ȥ��
//https://blog.csdn.net/yansmile1/article/details/46761271
//https://blog.csdn.net/patkritlee/article/details/53380419
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)  
{     
    int RemoveCount=0;       //��¼��ȥ�ĸ���  
    //��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
    Mat Pointlabel = Mat::zeros( Src.size(), CV_8UC1 );  
      
    if(CheckMode==1)  
    {  
        //cout<<"Mode: ȥ��С����. ";  
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
        //cout<<"Mode: ȥ���׶�. ";  
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
  
    vector<Point2i> NeihborPos;  //��¼�����λ��  
    NeihborPos.push_back(Point2i(-1, 0));  
    NeihborPos.push_back(Point2i(1, 0));  
    NeihborPos.push_back(Point2i(0, -1));  
    NeihborPos.push_back(Point2i(0, 1));  
    if (NeihborMode==1)  
    {  
        //cout<<"Neighbor mode: 8����."<<endl;  
        NeihborPos.push_back(Point2i(-1, -1));  
        NeihborPos.push_back(Point2i(-1, 1));  
        NeihborPos.push_back(Point2i(1, -1));  
        NeihborPos.push_back(Point2i(1, 1));  
    }  
    else cout<<"Neighbor mode: 4����."<<endl;  
    int NeihborCount=4+4*NeihborMode;  
    int CurrX=0, CurrY=0;  
    //��ʼ���  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 0)    
            {    
                //********��ʼ�õ㴦�ļ��**********  
                vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
                GrowBuffer.push_back( Point2i(j, i) );  
                Pointlabel.at<uchar>(i, j)=1;  
                int CheckResult=0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  
  
                for ( int z=0; z<GrowBuffer.size(); z++ )  
                {  
  
                    for (int q=0; q<NeihborCount; q++)                                      //����ĸ������  
                   {  
                        CurrX=GrowBuffer.at(z).x+NeihborPos.at(q).x;  
                        CurrY=GrowBuffer.at(z).y+NeihborPos.at(q).y;  
                        if (CurrX>=0&&CurrX<Src.cols&&CurrY>=0&&CurrY<Src.rows)  //��ֹԽ��  
                        {  
                            if ( Pointlabel.at<uchar>(CurrY, CurrX)==0 )  
                            {  
                                GrowBuffer.push_back( Point2i(CurrX, CurrY) );  //��������buffer  
                                Pointlabel.at<uchar>(CurrY, CurrX)=1;           //���������ļ���ǩ�������ظ����  
                            }  
                        }  
                    }  
  
                }  
                if (GrowBuffer.size()>AreaLimit) CheckResult=2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
                else {CheckResult=1;   RemoveCount++;}  
                for (int z=0; z<GrowBuffer.size(); z++)                         //����Label��¼  
                {  
                    CurrX=GrowBuffer.at(z).x;   
                    CurrY=GrowBuffer.at(z).y;  
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;  
                }  
                //********�����õ㴦�ļ��**********  
  
  
            }    
        }    
    }    
  
    CheckMode=255*(1-CheckMode);  
    //��ʼ��ת�����С������  
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

	//��������ͼ������ڻ���ͶӰͼ  (�ڵף�0 �ڣ�  1 ��)
    paintX = Mat::zeros( src.rows, src.cols, CV_8UC1 );       
	paintY = Mat::zeros( src.rows, src.cols, CV_8UC1 );

	//Mat paintX( src.cols, src.rows, CV_8UC1, Scalar( 0, 0, 0));
	//Mat paintY( src.cols, src.rows, CV_8UC1, Scalar( 0, 0, 0));
	//ת��Ϊ�Ҷ�ͼ��
	//cout<<"paintX.cols = "<<paintX.cols<<endl;
	//cout<<"paintX.rows = "<<paintX.rows<<endl;
	 
	
	//��ֵ��
	//��ʽ1
	preProcessing(src,src_binary,5);
	//��ʽ2
	/* Mat grayImg;
	cvtColor(src,grayImg,CV_RGB2GRAY);
	medianBlur(grayImg,grayImg,5);
	int blockSize = 25;  
    int constValue = 35;    
    adaptiveThreshold(grayImg, src_binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);*/
  
	

	//OPENCV��ֵ��ͼ���ڿ׶����/С����ȥ�� https://blog.csdn.net/yansmile1/article/details/46761271
	  double t = (double)getTickCount();  
	 char* OutPath = "�ֲ�_ȥ���׶�.jpg";  
    Mat Dst = Mat::zeros(src.size(), CV_8UC1);  
	 RemoveSmallRegion(src_binary, src_binary, 300, 1, 1);  //ȥ��С����
    //RemoveSmallRegion(src_binary, src_binary, 3000, 0, 0);  //���ն������Բ�����
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
         //����һ��ʵ��
	
	int i,j;
	float X_max=0,X_min=721, Y_max=0,Y_min=721;
	float X=0,Y=0;
   //��ֱ��������ۼӣ����֣�
	for( i=0; i<src_binary.cols; i++)           //��
	{
		for( j=0; j<src_binary.rows; j++)                //��
		{
			if( src_binary.at<uchar>( j, i ) == 255)                //ͳ�Ƶ��ǰ�ɫ���ص�����
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
	//���ƴ�ֱ�����ϵ�ͶӰ
	for( i=0; i<src_binary.cols; i++)
	{	
		for( j=0; j<v[i]; j++)
		{
			paintX.at<uchar>( j, i ) = 255;        //����ɫ������
			X=X+v[i];
		}
	}
	*X_diff= (X_max-X_min)/ X;



	//ˮƽ��������ۼӣ����֣�
	for( i=0; i<src_binary.rows; i++)           //��
	{
		for( j=0; j<src_binary.cols; j++)                //��
		{
			if( src_binary.at<uchar>( i, j ) == 255)       //ͳ�ư�ɫ���ص�����
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
	//����ˮƽ�����ϵ�ͶӰ
	for( i=0; i<src_binary.rows; i++)
	{	
		for( j=0; j<h[i]; j++)
		{
			paintY.at<uchar>( i, j ) = 255;        //���ɫ������
			Y=Y+h[i];
		}
	}
	 *Y_diff=(Y_max-Y_min)/Y;
      //printf("*X_diff=%f *Y_diff=%f \n",*X_diff,*Y_diff); 
	
	//��ʾͼ��
	//imshow("wnd_binary", src_binary);
	//imshow("wnd_X", paintX);
	//imshow("wnd_Y", paintY);
	
	


}

//------------------------------------------------------------------------------------------------------------------------------//
//--------------------------------------�����ѷ��ȼ��㷽ʽ���������б��---------------------------------------------------------------//

void measure_v(Mat &srcImg, double *maxdiff_width,double *mindiff_width)
{

	//...������ȣ��������ұ߽��ɫ���ص�������������,��Ϊ���ֻ࣬ɨ��ͬһ�е�����������������룩...//
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
					s[j]=j;//�˴���û�аѵ㼯�洢֮����ȡ�����Сֵ����Ϊ����ֶ��߳�����
					
					//if(k%2!=0)
					//{
					//	//printf("��һ��(%d,%d)\n",j,i);//û������һ��ӦΪ��ӡ����̫�࣬����ֻ����ʾһ���֣�vector(y,x)�洢�����Դ˴���j,i����ӡ
					//    p1.push_back(Point(j,i));
					//}
					//else if(k<=2) //**�˴�����Ϊelse if,�ο�ifǶ�����
					//{
					//	//printf("�ڶ���(%d,%d)\n",j,i); 
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
		
		//printf("��һ��(%d,%d)",min_j,i);//û������һ��ӦΪ��ӡ����̫�࣬����ֻ����ʾһ���֣�vector(y,x)�洢�����Դ˴���j,i����ӡ
		
					    p1.push_back(Point(min_j,i));
		}

		
		
		if((min_j!=srcImg.cols)&&(max_j!=0))
		
		{
		//printf("�ڶ���(%d,%d) ",max_j,i); 
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




//--------------------------------------������ȵ���ѷ������ѷ��Ϊ���࣬ÿһ��ÿһ��Ѱ������һ�����̾��룩����������ã�ʵ�ʲ���ô��-------------------------------------------------------------//
	//����������Ϊ��߽磬�򻭳�����С��Ӿ��ο��Ӱ���ȣ���ɫ�߿��к������߽��غϵĵط���������������Ϊʵ��Ӱ��
	
    //������ȡ�http://bbs.csdn.net/topics/391056710
	float p=0;
	float min_p=720;//ͼ���������ֵ
	vector<float> max_p;
	int q=0;
	int v=0;


	//��ʽ1 ȫ������
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
	//		 if(((6<p)&&p<min_p))//10<p�Ӵ�ӡ�����ݵó��ñ����ڽ�һ�������������ԭ���ǻ�����ʱ����������
	//			//if(p<min_p)

	//		 {
	//	      min_p=p;
	//        }
	//	}
	//	max_p.push_back(min_p);
 //     	
	//}

	//��ʽ2 ����10����

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
			 
			 //if(((6<p)&&p<min_p))//10<p�Ӵ�ӡ�����ݵó��ñ����ڽ�һ�������������ԭ���ǻ�����ʱ����������
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

	//...������ȣ��������ұ߽��ɫ���ص������������룩...//
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
			
			 
            cPointB1=srcImg.at<Vec3b>(j,i)[0];//**�˴��ر�ע�����Ϊ��rows��cols��
            cPointG1=srcImg.at<Vec3b>(j,i)[1];
            cPointR1=srcImg.at<Vec3b>(j,i)[2];
			if(cPointB1==0&cPointG1==0&cPointR1==255)
                {
					s[j]=j;//�˴���û�аѵ㼯�洢֮����ȡ�����Сֵ����Ϊ����ֶ��߳�����
					
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
		
		//printf("��һ��(%d,%d)",min_j,i);//û������һ��ӦΪ��ӡ����̫�࣬����ֻ����ʾһ���֣�vector(y,x)�洢�����Դ˴���j,i����ӡ
		
					    p1.push_back(Point(min_j,i));
		}

		
		
		if((min_j!=srcImg.rows)&&(max_j!=0))
		
		{
		//printf("�ڶ���(%d,%d ) ",max_j,i); 
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




//--------------------------------------������ȵ���ѷ������ѷ��Ϊ���࣬ÿһ��ÿһ��Ѱ������һ�����̾��룩����������ã�ʵ�ʲ���ô��-------------------------------------------------------------//
	
////������ȡ�http://bbs.csdn.net/topics/391056710
//����������Ϊ��߽磬�򻭳�����С��Ӿ��ο��Ӱ���ȣ���ɫ�߿��к������߽��غϵĵط���������������Ϊʵ��Ӱ��
	float p=0;
	float min_p=480;//ͼ���������ֵ
	vector<float> max_p;
	
	int q=0;
	int v=0;


	//��ʽ1 ȫ������
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
	//		 if(((6<p)&&p<min_p))//10<p�Ӵ�ӡ�����ݵó��ñ����ڽ�һ�������������ԭ���ǻ�����ʱ����������
	//			//if(p<min_p)

	//		 {
	//	      min_p=p;
	//        }
	//	}
	//	max_p.push_back(min_p);
 //     	
	//}


	//��ʽ2 ����10����
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
			 
			 //if(((6<p)&&p<min_p))//10<p�Ӵ�ӡ�����ݵó��ñ����ڽ�һ�������������ԭ���ǻ�����ʱ����������
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







