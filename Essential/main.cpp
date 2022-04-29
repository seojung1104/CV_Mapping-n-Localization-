
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <cpl_string.h>


////GDAL
#include "gdal_priv.h"
#include "cpl_string.h"

//
////Open GL
//#include "glew.h"
//#include "freeglut.h"


using namespace cv;
using namespace std;

Mat setRotation(double w, double p, double k);
Mat calcImgCoor(double x, double y);
Mat setPosition(double x, double y, double z);
Mat setBaseVector(double Bx, double By, double Bz);
Mat setIOP();
Mat setRT(Mat R, Mat P);
Mat setPixel(double px, double py);

Mat calcRO(Mat Rl,Mat Rr, Mat Pl, Mat Pr);
Mat calcEssential6(vector<Point2f> left, vector<Point2f> right);
Mat svd_eop(Mat essential, Mat re_eop, Mat ref_base);

Mat calcEssential3(double Bx, double By, double Bz, Mat Rl, Mat Rr);
Mat calcEssential4(double Bx, double By, double Bz, Mat Rr);
void svd_show_eop(Mat rota, Mat skew, Mat ref_eop);

double f = 18;//mm
double dp = 0.00479;
double c0 = 2464;
double r0 = 1632;
int main() {

	// 수사측 과제
	//이미지 read
	Mat left, right;
	left = imread("..\\..\\R0047624.jfif",IMREAD_COLOR);
	right = imread("..\\..\\R0047625.jfif", IMREAD_COLOR);
	//1. position Rotation;
	double Xl, Yl, Zl, wl, pl, kl;
	double Xr, Yr, Zr, wr, pr, kr;

	Xl = 292171.678; Yl = 4147334.36; Zl = 150.8;
	wl = 0.07400; pl = 0.01972; kl = -1.9980;

	Xr = 292154.207; Yr = 4147297.61; Zr = 150.1;
	wr = 0.08315; pr = -0.04533; kr = -2.01552;

	// 좌표값 대입
	Mat	Pl = setPosition(Xl, Yl,Zl);
	Mat Pr = setPosition(Xr, Yr, Zr);

	Mat Rl = setRotation(wl, pl, kl);
	Mat Rr = setRotation(wr, pr, kr);
	//2. Relative orientation
	//xl = 0; yl = 0; zl = 0;
	//wl = 0; pl = 0; kl = 0;

	double Bx, By, Bz;
	Bx = Xr - Xl; By = Yr - Yl; Bz = Zr - Zl;
	Mat ref_base = setBaseVector(Bx, By, Bz);
	Mat re_eop=calcRO(Rl,Rr,Pl,Pr);

	//3. Essential Matrix
	Mat E3=calcEssential3(Bx, By, Bz, Rl, Rr);
	//cout << E3<<endl;
	// 4. 2번 Relative wpk 사용
	double rw, rp, rk;
	Bx = re_eop.at<double>(0, 0);
	By = re_eop.at<double>(1, 0);
	Bz = re_eop.at<double>(2, 0);
	rw = re_eop.at<double>(3, 0);
	rp = re_eop.at<double>(4, 0);
	rk = re_eop.at<double>(5, 0);
	Mat re_Rr = Mat::zeros(3, 3, CV_64FC1);
	re_Rr.at<double>(0, 0) = re_eop.at<double>(0, 1);	re_Rr.at<double>(0, 1) = re_eop.at<double>(0, 2);	re_Rr.at<double>(0, 2) = re_eop.at<double>(0, 3);
	re_Rr.at<double>(1, 0) = re_eop.at<double>(1, 1);	re_Rr.at<double>(1, 1) = re_eop.at<double>(1, 2);	re_Rr.at<double>(1, 2) = re_eop.at<double>(1, 3);
	re_Rr.at<double>(2, 0) = re_eop.at<double>(2, 1);	re_Rr.at<double>(2, 1) = re_eop.at<double>(2, 2);	re_Rr.at<double>(2, 2) = re_eop.at<double>(2, 3);

	Mat E4 = calcEssential4(Bx, By, Bz, re_Rr);
	//cout << E4 << endl;
//	cout << E4 - E3 << endl;
	// 5.해결
	Mat ref_eop = Mat::zeros(6, 1, CV_64FC1);
	ref_eop.at<double>(0, 0)= Bx ;
	ref_eop.at<double>(1, 0)= By ;
	ref_eop.at<double>(2, 0)=Bz ;
	ref_eop.at<double>(3, 0) =rw ;
	ref_eop.at<double>(4, 0)=rp ;
	ref_eop.at<double>(5, 0)=rk ;
	svd_eop(E4, ref_eop, ref_base);

	// 6. tiepoint 측정후 Essential 추정
	vector<Point2f> left_point;
	vector<Point2f> right_point;
	left_point.push_back(Point2f(2103.0, 1450.0));
	left_point.push_back(Point2f(895, 746));
	left_point.push_back(Point2f(1352, 2489));
	left_point.push_back(Point2f(4558, 2538));
	left_point.push_back(Point2f(2452, 1713));
	left_point.push_back(Point2f(1129, 1470));
	left_point.push_back(Point2f(4093, 2483));
	left_point.push_back(Point2f(1852, 1572));
	left_point.push_back(Point2f(1277, 1874));

	right_point.push_back(Point2f(1195, 1665));
	right_point.push_back(Point2f(84, 1015));
	right_point.push_back(Point2f(487, 2688));
	right_point.push_back(Point2f(3656, 2723));
	right_point.push_back(Point2f(1524, 1915));
	right_point.push_back(Point2f(291, 1700));
	right_point.push_back(Point2f(3178, 2672));
	right_point.push_back(Point2f(960, 1786));
	right_point.push_back(Point2f(424, 2086));
	Mat essential666;
	essential666 = calcEssential6(left_point, right_point);
	cout << essential666 << endl;
//	return 0;
	return 0;
}
//3-2-1 system
Mat setRotation(double w, double p, double k) {
	Mat temp=Mat::zeros(3, 3, CV_64FC1);
	temp.at<double>(0, 0) = cos(p)*cos(k);							temp.at<double>(0, 1) = -cos(p)*sin(k);						 temp.at<double>(0,2) = sin(p);
	temp.at<double>(1, 0) = sin(w)*sin(p)*cos(k) + cos(w)*sin(k); temp.at<double>(1, 1) = cos(w)*cos(k) - sin(w)*sin(p)*sin(k); temp.at<double>(1, 2) = -sin(w)*cos(p);
	temp.at<double>(2, 0) = sin(w)*sin(k) - cos(w)*sin(p)*cos(k); temp.at<double>(2, 1) = sin(w)*cos(k) + cos(w)*sin(p)*sin(k); temp.at<double>(2, 2) = cos(w)*cos(p);

	return temp;
}
Mat calcImgCoor(double x, double y) {
	Mat temp = Mat::zeros(3, 1, CV_64FC1);
	temp.at<double>(0, 0) = (x-c0)*dp/f;
	temp.at<double>(1, 0) = (y-r0)*dp / f;
	temp.at<double>(2, 0) = 1;
	return temp;
}
Mat setBaseVector(double Bx, double By, double Bz) {
	Mat temp = Mat::zeros(3, 3, CV_64FC1);
	temp.at<double>(0, 0) = 0; temp.at<double>(0, 1) = -Bz; temp.at<double>(0, 2) = By;
	temp.at<double>(1, 0) = Bz; temp.at<double>(1, 1) = 0; temp.at<double>(1, 2) = -Bx;
	temp.at<double>(2, 0) = -By; temp.at<double>(2, 1) = Bx; temp.at<double>(2, 2) =0;
	return temp;
}
Mat setIOP() {
	Mat temp = Mat::zeros(3, 3, CV_64FC1);
	temp.at<double>(0, 0) = 1; temp.at<double>(0, 1) = 0; temp.at<double>(0, 2) = -c0;
	temp.at<double>(1, 0) = 0; temp.at<double>(1, 1) = 1; temp.at<double>(1, 2) = -r0;
	temp.at<double>(2, 0) = 0; temp.at<double>(2, 1) = 0; temp.at<double>(2, 2) = -f;
	return temp;
}
Mat setPixel(double px, double py) {
	Mat temp(3, 1, CV_64FC1);
	temp.at<double>(0, 0) = px;
	temp.at<double>(1, 0) = py;
	temp.at<double>(2, 0) = 1;
	return temp;
}
Mat calcEssential3(double Bx, double By, double Bz, Mat Rl, Mat Rr) {
	// Rl * B * Rr
	double base_scale = sqrt(Bx*Bx + By*By + Bz*Bz);
	Mat B = setBaseVector(Bx/ base_scale, By/ base_scale, Bz/ base_scale);
	Mat E = Rl.t()*B*Rr;
	return E;
}
Mat calcEssential4(double Bx, double By, double Bz, Mat Rr) {
	double base_scale = sqrt(Bx*Bx + By*By + Bz*Bz);
	Mat B = setBaseVector(Bx / base_scale, By / base_scale, Bz / base_scale);
	Mat E = B*Rr;
	return E;
}
Mat calcEssential6(vector<Point2f> left_p, vector<Point2f> right_p) {
	int unknownpara = 9;
	Mat A = Mat::zeros(left_p.size(), unknownpara, CV_64FC1);
	Mat L = Mat::zeros(unknownpara, 1, CV_64FC1);
	Mat temp = Mat::zeros(3, 3, CV_64FC1);;
	double F = dp / f;
	//for (int i = 0; i < left_p.size(); i++) {
	//	left_p[i].x = -(left_p[i].x - c0) * F;
	//	left_p[i].y = (left_p[i].y - r0) * F;
	//	right_p[i].x = -(right_p[i].x - c0) * F;
	//	right_p[i].y = (right_p[i].y - r0) * F;
	//}


	for (int i = 0 ;  i < left_p.size(); i++) {
		A.at<double>(i, 0) = left_p[i].x*right_p[i].x;
		A.at<double>(i, 1) = left_p[i].x *right_p[i].y;
		A.at<double>(i, 2) = left_p[i].x;
		A.at<double>(i, 3) = left_p[i].y * right_p[i].x ;
		A.at<double>(i, 4) = left_p[i].y *right_p[i].y;
		A.at<double>(i, 5) = left_p[i].y;
		A.at<double>(i, 6) = right_p[i].x;
		A.at<double>(i, 7) = right_p[i].y;
		A.at<double>(i, 8) = 1;
	}
	double e11, e12, e13, e21, e22, e23, e31, e32,e33;
	e11 = 1; e12 = 1; e13 = 1;
	e21 = 1; e22 = 1; e23 = 1;
	e31 = 1; e32 = 1; e33 = 1;
	Mat x = Mat::zeros(unknownpara, 1, CV_64FC1);
	int i = 0;
	while (1) {

		
		double de11, de12, de13, de21, de22, de23, de31, de32, de33;
		for (int i = 0; i < left_p.size(); i++) {
			L.at<double>(i, 0) = - (A.at<double>(i, 0)*e11+ A.at<double>(i, 1)*e12+A.at<double>(i, 2)*e13 
									+ A.at<double>(i, 3)*e21 + A.at<double>(i, 4)*e22 + A.at<double>(i, 5)*e23 
										+ A.at<double>(i, 6)*e31 + A.at<double>(i, 7)*e32+A.at<double>(i,8)*e33);
		}
		Mat kkk = A.t()*A;
		x = (kkk).inv()*A.t()*L;
		de11= x.at<double>(0, 0); de12 = x.at<double>(1, 0); de13 = x.at<double>(2, 0);
		de21 = x.at<double>(3, 0); de22 = x.at<double>(4, 0); de23 = x.at<double>(5, 0);
		de31 = x.at<double>(6, 0); de32 = x.at<double>(7, 0); de33 = x.at<double>(8, 0);

		e11 += de11; e12 += de12; e13 += de13;
		e21 += de21; e22 += de22; e23 += de23;
		e31 += de31; e32 += de32; e33 += de33;

		if (abs(de11) < 0.000001 && abs(de21) < 0.000001 && abs(de31) < 0.000001 && abs(de12) < 0.000001 && abs(de22) < 0.000001 && abs(de23) < 0.000001 && abs(de13) < 0.000001 && abs(de23) < 0.000001) {
			break;
		}
		temp.at<double>(0, 0)= e11; temp.at<double>(0, 1) = e12; temp.at<double>(0, 2) = e13;
		temp.at<double>(1, 0) = e21; temp.at<double>(1, 1) = e22; temp.at<double>(1, 2) = e23;
		temp.at<double>(2, 0) = e31; temp.at<double>(2, 1) = e32; temp.at<double>(2, 2) = 1;
	/*	if (i == 100) {
			break;
		}
		i++;*/
	}

	return temp;
}
Mat setPosition(double x, double y, double z) {
	Mat temp = Mat::zeros(3, 1, CV_64FC1);
	temp.at<double>(0, 0) = x;
	temp.at<double>(1, 0) = y;
	temp.at<double>(2, 0) = z;
	return temp;
}
Mat calcRO(Mat Rl, Mat Rr, Mat Pl, Mat Pr) {

	//Pr = Pr-Pl;
	//Pl = setPosition(0, 0, 0);
	Mat Lrt = setRT(Rl, Pl); // [R | T]
	Mat Rrt = setRT(Rr, Pr); //
	Mat Result = Lrt.inv()*Rrt;
	double scale = sqrt(Result.at<double>(0, 3)* Result.at<double>(0, 3) + Result.at<double>(1, 3)*Result.at<double>(1, 3) + Result.at<double>(2, 3)*Result.at<double>(2, 3));
	Result.at<double>(0, 3) = Result.at<double>(0, 3)/scale;
	Result.at<double>(1, 3) = Result.at<double>(1, 3) / scale;
	Result.at<double>(2, 3) = Result.at<double>(2, 3) / scale;
	double sinp,phi,k,w;
	sinp = Result.at<double>(0, 2);
	phi = asin(sinp);
	k = acos(Result.at<double>(0, 0) / (sqrt(1 - sinp*sinp)));
	w = acos(Result.at<double>(2, 2) / (sqrt(1 - sinp*sinp)));
	Mat temp = Mat::zeros(6, 4, CV_64FC1);
	temp.at<double>(0, 0) = Result.at<double>(0,3);
	temp.at<double>(1, 0) = Result.at<double>(1, 3);
	temp.at<double>(2, 0) = Result.at<double>(2, 3);
	temp.at<double>(3, 0) = w;
	temp.at<double>(4, 0) = phi;
	temp.at<double>(5, 0) = k;
	temp.at<double>(0, 1) = Result.at<double>(0, 0); 	temp.at<double>(0, 2) = Result.at<double>(0, 1); 	temp.at<double>(0, 3) = Result.at<double>(0, 2);
	temp.at<double>(1, 1) = Result.at<double>(1, 0);	temp.at<double>(1, 2) = Result.at<double>(1, 1);	temp.at<double>(1, 3) = Result.at<double>(1, 2);
	temp.at<double>(2, 1) = Result.at<double>(2, 0);	temp.at<double>(2, 2) = Result.at<double>(2, 1);	temp.at<double>(2, 3) = Result.at<double>(2, 2);

	return temp;
}
Mat setRT(Mat R, Mat P) {
	Mat temp = Mat::zeros(4, 4, CV_64FC1);
	temp.at<double>(0, 0) = R.at<double>(0, 0); temp.at<double>(0, 1) = R.at<double>(0, 1); temp.at<double>(0, 2) = R.at<double>(0, 2); temp.at<double>(0, 3) = P.at<double>(0, 0);
	temp.at<double>(1, 0) = R.at<double>(1, 0); temp.at<double>(1, 1) = R.at<double>(1, 1); temp.at<double>(1, 2) = R.at<double>(1, 2); temp.at<double>(1, 3) = P.at<double>(1, 0);
	temp.at<double>(2, 0) = R.at<double>(2, 0); temp.at<double>(2, 1) = R.at<double>(2, 1); temp.at<double>(2, 2) = R.at<double>(2, 2); temp.at<double>(2, 3) = P.at<double>(2, 0);
	temp.at<double>(3, 0) =0;					temp.at<double>(3, 1) = 0;					temp.at<double>(3, 2) = 0;					temp.at<double>(3, 3) = 1;
	return temp;
}
Mat svd_eop(Mat essential, Mat ref_eop, Mat ref_base) {
	Mat temp,W,U,Vt;
	SVD::compute(essential, W, U, Vt);
	Mat z= Mat::zeros(3, 3, CV_64FC1);
	z.at<double>(0, 1) = 1;
	z.at<double>(1, 0) = -1;
	Mat w = Mat::zeros(3, 3, CV_64FC1);
	w.at<double>(0, 1) = -1;
	w.at<double>(1, 0) = 1;
	w.at<double>(2, 2) = 1;

	Mat d_skew = Mat::zeros(3, 3, CV_64FC1);
	Mat d_Rota = Mat::zeros(3, 3, CV_64FC1);

	Mat skew1 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota1 = Mat::zeros(3, 3, CV_64FC1);

	skew1 = U*z*U.t();
	Rota1 = U*w.t()*Vt;
	svd_show_eop(Rota1, skew1, ref_eop);
	// uwtvt | u3

	Mat skew2 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota2 = Mat::zeros(3, 3, CV_64FC1);

	skew2 = -U*z*U.t();
	Rota2 = U*w.t()*Vt; 
	svd_show_eop(Rota2, skew2, ref_eop);

	// uwtvt | -u3

	Mat skew3 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota3 = Mat::zeros(3, 3, CV_64FC1);
	
	skew3 = U*z*U.t();
	Rota3 = U*w*Vt;
	svd_show_eop(Rota3, skew3, ref_eop);
	// uwvt | -u3

	Mat skew4 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota4 = Mat::zeros(3, 3, CV_64FC1);

	skew4 = -U*z*U.t();
	Rota4 = U*w*Vt;
	svd_show_eop(Rota4, skew4, ref_eop);
	 // uwvt | u3

	Mat skew5 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota5 = Mat::zeros(3, 3, CV_64FC1);

	skew5 = U*z*U.t();
	Rota5 = -U*w.t()*Vt;
	svd_show_eop(Rota5, skew5, ref_eop);
	// uwtvt | u3

	Mat skew6 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota6 = Mat::zeros(3, 3, CV_64FC1);

	skew6 = -U*z*U.t();
	Rota6 =-U*w.t()*Vt;
	svd_show_eop(Rota6, skew6, ref_eop);
	// uwtvt | -u3

	Mat skew7 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota7 = Mat::zeros(3, 3, CV_64FC1);

	skew7 = U*z*U.t();
	Rota7 = -U*w*Vt;
	svd_show_eop(Rota7, skew7, ref_eop);
	// uwvt | -u3

	Mat skew8 = Mat::zeros(3, 3, CV_64FC1);
	Mat Rota8 = Mat::zeros(3, 3, CV_64FC1);

	skew8 = -U*z*U.t();
	Rota8 = -U*w*Vt;
	svd_show_eop(Rota8, skew8, ref_eop);
	// uwvt | u3

	return  temp;
}
void svd_show_eop(Mat rota, Mat skew,Mat ref_eop) {
	Mat temp = Mat::zeros(6, 1, CV_64FC1);
	double sinp, phi, k, w;
	sinp = rota.at<double>(0, 2);
	phi = asin(sinp);
	k = acos(rota.at<double>(0, 0) / (sqrt(1 - sinp*sinp)));
	w = acos(rota.at<double>(2, 2) / (sqrt(1 - sinp*sinp)));
	temp.at<double>(0, 0) = skew.at<double>(2, 1);
	temp.at<double>(1, 0) = skew.at<double>(0, 2);
	temp.at<double>(2, 0) = skew.at<double>(1, 0);
	temp.at<double>(3, 0) = w;
	temp.at<double>(4, 0) = phi;
	temp.at<double>(5, 0) = k;
	cout << ref_eop-temp << endl;
}