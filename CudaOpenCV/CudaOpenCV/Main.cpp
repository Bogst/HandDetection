//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;
void addWithCuda(int *a, int *b, int rows, int cols);
void Detection(bool &calibration, Mat &frame, Mat &frameh, Mat &bgframe, int &crlowerb, int &crupperb, int &cblowerb, int &cbupperb);
void thinning(cv::Mat& im);


#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctime>
#include <cstdio>

using namespace cv;
using namespace std;

//Mat img;
int crlowerb = 150, crupperb = 180, cblowerb = 60, cbupperb=110;


Mat frame;



std::clock_t start;


//
int main()
{




	//cv::Mat src = cv::imread("C:\\Users\\OrderAdChaos\\Desktop\\Untitled.png");
	//if (src.empty())
	//	return -1;

	//cv::Mat bw;
	//cv::cvtColor(src, bw, CV_BGR2GRAY);
	//cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

	//thinning(bw);

	//cv::imshow("src", src);
	//cv::imshow("dst", bw);
	//cv::waitKey(0);








	
	//int  cblowerb = 110, cbupperb = 128; //crlowerb , crupperb
	int crlowerb1 = 142, crupperb1 = 160, cblowerb1 = 99, cbupperb1 = 107;
//	VideoCapture capTel(2);

	VideoCapture cap(0);

//	VideoCapture cap;
//	VideoCapture capTel(0);
	//cap.open("http://192.168.0.108:8080/video");
	//capTel.open("http://192.168.0.108:8080/video");
	//cap.open("http://192.168.0.102:8080/video");

	if (!cap.isOpened()) 
		return -1;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	Mat bgframe;
	Mat bgframe1;
	bool calibration = true;
	if (calibration)
	{
		cap >> bgframe;
		cap >> bgframe1;
	}

	while (true)
	{
		start = std::clock();
		Mat frameh, frame2,frameh2;//frame
		frameh = Mat::zeros(256, 256, 4);
		frameh2 = Mat::zeros(256, 256, 4);
		cap >> frame;
//		capTel >> frame2;
		medianBlur(frame, frame, 5);
		medianBlur(frame2, frame2, 5);
//		Detection(calibration, frame2, frameh2, bgframe1, crlowerb1, crupperb1, cblowerb1, cbupperb1);

		cout << "Time before detection"<<(std::clock() - start) / (double)CLOCKS_PER_SEC<<endl;
		Detection(calibration, frame, frameh, bgframe, crlowerb, crupperb, cblowerb, cbupperb);


//		namedWindow("Image", 1);
//		imshow("Image", frame);

		namedWindow("Connected Components", 1);
		createTrackbar("Red Lower", "Connected Components", &crlowerb, 250, 0);
		createTrackbar("Red Upper", "Connected Components", &crupperb, 250, 0);
		createTrackbar("Blue Lower", "Connected Components", &cblowerb, 250, 0);
		createTrackbar("Blue Upper", "Connected Components", &cbupperb, 250, 0);
		imshow("Connected Components", frame);

//		imshow("webcam", frame);
//		imshow("histogram", frameh);
//		imshow("tel", frame2);
//		imshow("htel", frameh2);
		//system("pause");

		if (waitKey(30) >= 0) break;
		system("cls");
	}

	return 0;
}


void Detection(bool &calibration, Mat &frame, Mat &frameh, Mat &bgframe, int &crlowerb, int &crupperb, int &cblowerb, int &cbupperb)
{

	uchar* test = (uchar*)malloc(sizeof(frame.data));
	test = frame.data;
	int* mask;
	mask = (int*)malloc(frame.rows*frame.cols*sizeof(int));


	int* pixels;
	pixels = (int*)malloc(frame.rows*frame.cols*sizeof(int));



	int max = 0;
	bool first = true;
	if (calibration)
	{
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				//cout << (int)frame.data[i*frame.cols + j]<<" ";
				//frame.data[(i*frame.cols)*3 + j] = 255 - (int)frame.data[(i*frame.cols)*3 + j];
				Vec3b intensity = bgframe.at<Vec3b>(i, j);

				int y, cr, cb;
				y = (int)(0.299   * intensity.val[2] + 0.587   * intensity.val[1] + 0.114   * intensity.val[0]);
				cr = (int)(0.50000 * intensity.val[2] - 0.41869 * intensity.val[1] - 0.08131 * intensity.val[0]) + 127;
				cb = (int)(-0.16874 * intensity.val[2] - 0.33126 * intensity.val[1] + 0.50000 * intensity.val[0]) + 127;
				intensity.val[0] = y;
				intensity.val[1] = cr;
				intensity.val[2] = cb;
				bgframe.at<Vec3b>(i, j) = intensity;
			}
		}
	}
	if (calibration)
	{
		system("pause");
		calibration = false;
	}

	start = std::clock();
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			//cout << (int)frame.data[i*frame.cols + j]<<" ";
			//frame.data[(i*frame.cols)*3 + j] = 255 - (int)frame.data[(i*frame.cols)*3 + j];
			Vec3b intensity = frame.at<Vec3b>(i, j);

			/* b */			//intensity.val[0] = 78;					
			/* g */			//intensity.val[1] = 76;
			/* r */			//intensity.val[2] = 88;
			int y, cr, cb;
			y = (int)(0.299   * intensity.val[2] + 0.587   * intensity.val[1] + 0.114   * intensity.val[0]);
			cr = (int)(0.50000 * intensity.val[2] - 0.41869 * intensity.val[1] - 0.08131 * intensity.val[0]) + 127;
			cb = (int)(-0.16874 * intensity.val[2] - 0.33126 * intensity.val[1] + 0.50000 * intensity.val[0]) + 127;
			//				cout << (int)intensity.val[0] << " " << (int)intensity.val[1] << " " << (int)intensity.val[2] << endl << y <<" "<< cb << " " << cr << endl;
			//intensity.val[0] = y;
			//intensity.val[1] = cr;
			//intensity.val[2] = cb;
			Vec3b histo = frameh.at<Vec3b>(cr, cb);
			histo.val[0] = histo.val[0] + 1;
			Vec3b bgvalues = bgframe.at<Vec3b>(i, j);
			if ((bgvalues.val[1] - cr > 8 || bgvalues.val[2] - cb > 8 || true) && ((cr>crlowerb && cr< crupperb&&cb>cblowerb && cb< cbupperb)))
			{
				mask[i*frame.cols + j] = 1;
			}
			else
			{
				mask[i*frame.cols + j] = 0;
			}

			//if (cr<crlowerb || cr> crupperb || cb<cblowerb || cb> cbupperb)
			//{
			//	mask[i*frame.cols+j] = 0;
			////	intensity.val[0] = 0;
			////	intensity.val[1] = 0;
			////	intensity.val[2] = 0;
			//}
			//else
			//{

			//	mask[i*frame.cols + j] = 1;
			//}

			//if (cr<130 || cr> 160)
			//{
			//	intensity.val[0] = 0;
			//	intensity.val[1] = 0;
			//	intensity.val[2] = 0;
			//}
			//if (cb<95 || cb> 125)
			//{
			//	intensity.val[0] = 0;
			//	intensity.val[1] = 0;
			//	intensity.val[2] = 0;
			//}

			if (max < histo.val[0])
			{
				max = histo.val[0];
			}
			frameh.at<Vec3b>(cr, cb) = histo;
			//cout << (int)histo.val[0];
			frame.at<Vec3b>(i, j) = intensity;
		}


		//DilatareKernel<<<16,16>>>(&frame, &frame1);	

	}

	cout << "Time transforming into ycrcb" << (std::clock() - start) / (double)CLOCKS_PER_SEC<<endl;
	start = std::clock();
	addWithCuda(mask, pixels, (int)frame.rows, (int)frame.cols);
	cout << "Time in cuda: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << endl;
	start = std::clock();

	int lx=frame.cols, hx=0, ly=frame.rows, hy = 0;
	
	//frame.copyTo(matpxls);
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (pixels[i*frame.cols+ j] == 1)
			{
				if (j < lx)
				{
					lx = j;
				}
				if (j> hx)
				{
					hx = j;
				}
				if (i < ly)
				{
					ly = i;
				}if (i > hy)
				{
					hy = i;
				}
			}
		}
	}
	Mat matpxls(1,1,4);
	if (hx - lx > 5&& hy - ly > 5)
	{
		matpxls=Mat(hx - lx+10, hy - ly+10, 4);
		for (int i = lx; i < hx; i++)
		{
			for (int j = ly; j < hy; j++)
			{
				matpxls.at<int>(i-lx, j-ly) = pixels[j*frame.cols + i];
			}
		}
//		cout << "Pregatire scheleare: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << endl;
//		start = std::clock();
//		thinning(matpxls);
//		cout << "Scheletare propriuzisa: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << endl;
//		start = std::clock();
//		matpxls *= 255;
//		imshow("tesat", matpxls);
	}
	


	//cout << frame.rows << endl;
	//cout << pixels[0]<<endl;
	//cout << ((frame.rows - 1) / TILE_WIDTH + 1);
	//cout << cudaGetErrorString(cudaStatus);


	//for (int i = 0; i < frame.rows; i++)
	//{
	//	//for (int j = 0; j < frame.cols; j++)
	//	//{
	//		cout << pixels[i*frame.cols ] << " ";
	//	//}
	//	cout << endl;
	//}

	//for (int i = 0; i < 255; i++)
	//{
	//	for (int j = 0; j < 255; j++)
	//	{
	//		Vec3b histo = frameh.at<Vec3b>(i, j);
	//		if (i == crlowerb || i == crupperb || j == cblowerb || j == cbupperb)
	//			histo.val[0] = 255;
	//		else
	//			histo.val[0] = (int)(histo.val[0] / (double)(max)* 255);
	//		frameh.at<Vec3b>(i, j) = histo;
	//	}
	//}
	//int *projectionX, *projectionY;
	//projectionX = (int*)malloc(frame.cols*sizeof(int));
	//projectionY = (int*)malloc(frame.rows*sizeof(int));
	//for (int i = 0; i < frame.cols; i++)
	//{
	//	projectionX[i] = 0;
	//}
	//for (int i = 0; i < frame.rows; i++)
	//{
	//	projectionY[i] = 0;
	//}
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (pixels[i*frame.cols+j] == 0)
			{

				Vec3b channels = frame.at<Vec3b>(i, j);
				//channels.val[0] = 0;
				//channels.val[1] = 0;
				//channels.val[2] = 0;
				frame.at<Vec3b>(i, j) = channels;
			}
			else
			{
				
				//projectionX[j]++;
				//projectionY[i]++;
				Vec3b channels = frame.at<Vec3b>(i, j);
				channels.val[0] = 0;
				channels.val[1] = 0;
				channels.val[2] = 255 ;
				frame.at<Vec3b>(i, j) = channels;
				if (i>ly && i< hy&&  j>lx&&j<hx && matpxls.at<int>(j - lx, i - ly) != 0)
				{
					Vec3b channels = frame.at<Vec3b>(i, j);
					channels.val[0] = 255;
					channels.val[1] = 0;
					channels.val[2] = 0;
					frame.at<Vec3b>(i, j) = channels;
				}
			}
		}
	}
	cout << "Time for show: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << endl;
	start = std::clock();
	/*int centerX = 0, centerY = 0,centerXpos=50, centerYpos=50;
	for (int i = 0; i < frame.cols; i++)
	{
		if (projectionX[i]>centerX)
		{
			centerX = projectionX[i];
			centerXpos = i;
		}
	}
	for (int i = 0; i < frame.rows; i++)
	{
		if (projectionY[i]>centerY)
		{
			centerY = projectionY[i];
			centerYpos = i;
		}
	}
	Vec3b centerFr = frame.at<Vec3b>(centerYpos, centerXpos);
	centerFr.val[0] = 255 - centerFr.val[0];
	centerFr.val[1] = 255 - centerFr.val[1];
	centerFr.val[2] = 255 - centerFr.val[2];
	frame.at<Vec3b>(centerYpos, centerXpos) = centerFr;
	for (int i = 0; i < 255; i +=10)
	{
		Vec3b histo = frameh.at<Vec3b>(i, 1);
		histo.val[0] = 255;
		frameh.at<Vec3b>(i, 1) = histo;
		histo = frameh.at<Vec3b>(1, i);
		histo.val[0] = 255;
		frameh.at<Vec3b>(1, i) = histo;
	}*/
	delete mask;
	delete pixels;
}















void thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.rows, im.cols, 4);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			int p2 = im.at<int>(i - 1, j);
			int p3 = im.at<int>(i - 1, j + 1);
			int p4 = im.at<int>(i, j + 1);
			int p5 = im.at<int>(i + 1, j + 1);
			int p6 = im.at<int>(i + 1, j);
			int p7 = im.at<int>(i + 1, j - 1);
			int p8 = im.at<int>(i, j - 1);
			int p9 = im.at<int>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
			{
				marker.at<int>(i, j) = 1;

			}
				
		}
	}

	im &= ~marker;
}

/**
* Function for thinning the given binary image
*
* @param  im  Binary image with range = 0-255
*/
void thinning(cv::Mat& im)
{
	
//	im /= 255;

	cv::Mat prev(im.rows, im.cols, 4);
	prev = cv::Mat::zeros(im.rows, im.cols, 4);
	cv::Mat diff;
	bool change;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
		

	//	imshow("test",im);

	} while (cv::countNonZero(diff) > 0);

//	im *= 255;
}






















double findDistance(int distanceBetweenCams, int widthResolution, int xA, int yA, int xB, int yB, bool unidimensional = true)
{
	if (unidimensional)
	{
		yA = 0;
		yB = 0;
	}
	double distBetweenObj = sqrt((xA - xB)*(xA - xB) + (yA - yB)*(yA - yB));
	return (double)distanceBetweenCams / (double)distBetweenObj / (double)widthResolution;
}
