#define _USE_MATH_DEFINES
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;

//direct translation to C++ from MATLAB source file: cgm.technion.ac.il/people/Viki/figure1&5_cluster/computeColor.m
Mat make_color_wheel() {

	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int num_cols = RY + YG + GC + CB + BM + MR;
	int col = 0;

	Mat color_wheel = Mat::zeros(num_cols, 3, CV_64FC1);

	//RY calculation
	for (int i = 0; i < RY; i++) {
		color_wheel.at<double>(i, 0) = 255;
		color_wheel.at<double>(i, 1) = floor(255 * i / RY);
	}
	col += RY;

	//YG calculation
	for (int i = 0; i < YG; i++) {
		color_wheel.at<double>(col + i, 0) = 255 - floor(255 * i / YG);
		color_wheel.at<double>(col + i, 1) = 255;
	}
	col += YG;

	//GC calculation
	for (int i = 0; i < GC; i++) {
		color_wheel.at<double>(col + i, 1) = 255;
		color_wheel.at<double>(col + i, 2) = floor(255 * i / GC);
	}
	col += GC;

	//CB calculation
	for (int i = 0; i < CB; i++) {
		color_wheel.at<double>(col + i, 1) = 255 - floor(255 * i / CB);
		color_wheel.at<double>(col + i, 2) = 255;
	}
	col += CB;

	//BM calculation
	for (int i = 0; i < BM; i++) {
		color_wheel.at<double>(col + i, 2) = 255;
		color_wheel.at<double>(col + i, 0) = floor(255 * i / BM);
	}
	col += BM;

	//MR calculation
	for (int i = 0; i < MR; i++) {
		color_wheel.at<double>(col + i, 2) = 255 - floor(255 * i / MR);
		color_wheel.at<double>(col + i, 0) = 255;
	}

	return color_wheel;
}

//indirect translation to C++ from MATLAB source file: cgm.technion.ac.il/people/Viki/figure1&5_cluster/computeColor.m
Mat compute_color(Mat U, Mat V) {

	Mat img;
	Mat color_wheel = make_color_wheel();
	int num_cols = color_wheel.rows;

	Mat U_squared, V_squared, rad;
	cv::pow(U, 2, U_squared);
	cv::pow(V, 2, V_squared);
	cv::sqrt(U_squared + V_squared, rad);

	Mat a = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			double v_element = V.at<double>(i, j);
			double u_element = U.at<double>(i, j);
			a.at<double>(i, j) = atan2(-v_element, -u_element) / M_PI;
		}
	}

	Mat fk = (a + 1) / 2 * (num_cols - 1); //remove +1 for C++ indexing
	Mat k0 = Mat::zeros(U.rows, U.cols, CV_64FC1);

	for (int i = 0; i < k0.rows; i++) {
		for (int j = 0; j < k0.cols; j++) {
			k0.at<double>(i, j) = floor(fk.at<double>(i, j));
		}
	}

	Mat k1 = k0 + 1;

	for (int i = 0; i < k1.rows; i++) {
		for (int j = 0; j < k1.cols; j++) {
			if (k1.at<double>(i, j) == num_cols) //adjust for overflow in indexing
				k1.at<double>(i, j) = 0;
		}
	}

	Mat f = fk - k0;
	Mat f_prime = 1 - f;

	vector<cv::Mat> channels; //to store the RGB channels

	for (int i = 0; i < color_wheel.cols; i++) {
		Mat col0 = Mat::zeros(k0.rows, k0.cols, CV_64FC1);
		Mat col1 = Mat::zeros(k1.rows, k1.cols, CV_64FC1);

		for (int j = 0; j < k0.rows; j++) {
			for (int k = 0; k < k0.cols; k++) {

				double col0_index = k0.at<double>(j, k);
				col0.at<double>(j, k) = color_wheel.at<double>(col0_index, i) / 255.0;

				double col1_index = k1.at<double>(j, k);
				col1.at<double>(j, k) = color_wheel.at<double>(col1_index, i) / 255.0;
			}
		}

		Mat col_first, col_second, col;

		multiply(f_prime, col0, col_first);
		multiply(f, col1, col_second);

		col = col_first + col_second;

		for (int l = 0; l < col.rows; l++) {
			for (int m = 0; m < col.cols; m++) {
				if (rad.at<double>(l, m) <= 1) {
					double col_val = 1 - rad.at<double>(l, m) * (1 - col.at<double>(l, m));
					col.at<double>(l, m) = col_val;
				}
				else
					col.at<double>(l, m) *= 0.75;
			}
		}
		channels.push_back(col);
	}
	reverse(channels.begin(), channels.end());
	cv::merge(channels, img);

	return img;
}


void optical_flow_analysis(Mat mat1, Mat mat2, int iterations = 100, int avg_window = 5, double alpha = 1) {

	//C++ time elapsed: stackoverflow.com/questions/2808398/easily-measure-elapsed-time
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	Mat img_gray, img2_gray;

	//convert to grayscale; ints from 0 to 255
	cvtColor(mat1, img_gray, CV_BGR2GRAY);
	cvtColor(mat2, img2_gray, CV_BGR2GRAY);

	//convert to double precision floats
	Mat img_gray_db, img2_gray_db;

	//CV_64FC1 for single channel
	img_gray.convertTo(img_gray_db, CV_64FC1, 1.0 / 255.0);
	img2_gray.convertTo(img2_gray_db, CV_64FC1, 1.0 / 255.0);

	//calculate directional gradients of first image
	Mat I_t = img2_gray_db - img_gray_db;

	/*cout << img_gray_db.at<double>(0, 0) << endl;
	cout << img2_gray_db.at<double>(0, 0) << endl;
	cout << I_t.at<double>(0, 0) << endl;

	cout << "Rows in img_gray_db is " << img_gray_db.rows << " and cols is " << img_gray_db.cols << endl;
	cout << "Rows in img2_gray_db is " << img2_gray_db.rows << " and cols is " << img2_gray_db.cols << endl;
	cout << "Rows in I_t is " << I_t.rows << " and cols is " << I_t.cols << endl;*/

	//calculate equivalent of imgradientxy for img_gray_db

	Mat I_x, I_y;
	int ddepth = -1; //outputs same depth as input

	Sobel(img_gray_db, I_x, ddepth, 1, 0, 3); //X gradient
	Sobel(img_gray_db, I_y, ddepth, 0, 1, 3); //Y gradient

	//initialize zero-filled matrices
	Mat U = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);
	Mat V = Mat::zeros(I_t.rows, I_t.cols, CV_64FC1);

	//initialize kernel
	Mat kernel = Mat::ones(avg_window, avg_window, CV_64FC1) / pow(avg_window, 2);

	//run multiple iterations to get horizontal and vertical flow

	for (int i = 0; i < iterations; i++) {
		
		Mat U_avg, V_avg;

		// from stackoverflow.com/questions/10309561/is-there-any-function-in-opencv-which-is-equivalent-to-matlab-conv2 :
		//perform 2D convolutions equivalent to "same" argument in MATLAB
		Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);

		//no need to flip kernel because it's symmetric
		filter2D(U, U_avg, U.depth(), kernel, anchor, 0, BORDER_CONSTANT);
		filter2D(V, V_avg, V.depth(), kernel, anchor, 0, BORDER_CONSTANT);

		//update U and V
		Mat C_prod1, C_prod2, I_x_squared, I_y_squared, I_x_C, I_y_C, C;

		multiply(I_x, U_avg, C_prod1);
		multiply(I_y, V_avg, C_prod2);
		multiply(I_x, I_x, I_x_squared);
		multiply(I_y, I_y, I_y_squared);

		Mat C_num = C_prod1 + C_prod2 + I_t; 
		Mat C_den = pow(alpha, 2) + I_x_squared + I_y_squared;
		
		divide(C_num, C_den, C);

		multiply(I_x, C, I_x_C);
		multiply(I_y, C, I_y_C);

		U = U_avg - I_x_C;
		V = V_avg - I_y_C;
	}

	//compute color equivalence

	Mat img = compute_color(U, V);


	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	cout << "The time elapsed is " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << endl;

	string window1 = "First image";
	string window2 = "Second image";
	string window3 = "Third image";

	namedWindow(window1, WINDOW_AUTOSIZE);
	imshow(window1, U);

	namedWindow(window2, WINDOW_AUTOSIZE);
	imshow(window2, V);

	namedWindow(window3, WINDOW_AUTOSIZE);
	imshow(window3, img);

	waitKey(0);
}

int main(int argc, char* argv[]) {
	//read in images
	Mat img = imread(argv[1]);
	Mat img2 = imread(argv[2]);

	//if files don't load, exit
	if (img.empty() || img2.empty())
		return -1;

	//call optical flow analysis
	optical_flow_analysis(img, img2);

	return 0;
}
