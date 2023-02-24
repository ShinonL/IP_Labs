// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage() {
	char fname[MAX_PATH];

	// openFileDialog deschide un browse for file
	while(openFileDlg(fname)) {
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();

		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);

		waitKey();
	}
}

void changeGrayScale(/*int number*/) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				//int lighterValue = val;
				//if (number >= 0)
					//lighterValue = min(255, val + number);
				//else lighterValue = max(255, val - number);
				int lighterValue = min(255, val + 50);
				dst.at<uchar>(i, j) = lighterValue;
			}

		imshow("Initial image", src);
		imshow("Lighter image", dst);

		waitKey();
	}
}

void createFourColorsImage() {
	int height = 400;
	int width = 400;
	Mat_<Vec3b> newImage(height, width, CV_8UC3);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			if (i < height / 2 && j < width / 2)
				newImage(i, j) = Vec3b(5, 22, 37);
			if (i < height / 2 && j >= width / 2)
				newImage(i, j) = Vec3b(167, 219, 247);
			if (i >= height / 2 && j < width / 2)
				newImage(i, j) = Vec3b(132, 121, 232);
			if (i >= height / 2 && j >= width / 2)
				newImage(i, j) = Vec3b(183, 175, 156);
		}

	imshow("Initial image", newImage);
	waitKey();
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence() {
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame)) {
		Mat grayFrame;

		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

		Canny(grayFrame,edges,40,100,3);

		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);

		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap() {
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) { // openenig the video device failed
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;) {
		cap >> frame; // get a new frame from camera
		if (frame.empty()) {
			printf("End of the video file\n");
			break;
		}

		frameNum++;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}

		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;

			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");

			bool bSuccess = imwrite(fileName, frame);

			if (!bSuccess) {
				printf("Error writing the snapped image\n");
			} else imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param) {
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == EVENT_LBUTTONDOWN) {
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int hist_cols, const int hist_height) {
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;

	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void inverseFloatMatrix() {
	float values[9] = { 1, 2, 3, 4, 51, 6, 7, 8, 11 };
	Mat M(3, 3, CV_32FC1, values); //4 parameter constructor

	std::cout << M.inv() << std::endl;

	getchar();
	getchar();
}

void splitImage() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat rChannel = Mat(height, width, CV_8UC3);
		Mat gChannel = Mat(height, width, CV_8UC3);
		Mat bChannel = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++) 
			for (int j = 0; j < width; j++) {
				Vec3b color = src.at<Vec3b>(i, j);

				rChannel.at<Vec3b>(i, j) = Vec3b(0, 0, color[2]);
				gChannel.at<Vec3b>(i, j) = Vec3b(0, color[1], 0);
				bChannel.at<Vec3b>(i, j) = Vec3b(color[0], 0, 0);
			}

		imshow("Initial Image", src);
		imshow("Red Image", rChannel);
		imshow("Green Image", gChannel);
		imshow("Blue Image", bChannel);
		waitKey();
	}
}

void rgbToGrayscale() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat grayScaleImg = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b color = src.at<Vec3b>(i, j);

				grayScaleImg.at<uchar>(i, j) = (color[0] + color[1] + color[2]) / 3;
			}

		imshow("Initial Image", src);
		imshow("GrayScale Image", grayScaleImg);
		waitKey();
	}
}

void grayscaleToBW() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat grayScaleImg = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				grayScaleImg.at<uchar>(i, j) = (src.at<uchar>(i, j) < 125) ? 0 : 255;
			}

		imshow("Initial Image", src);
		imshow("GrayScale Image", grayScaleImg);
		waitKey();
	}
}

void rgbToHSV() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat hImage = Mat(height, width, CV_8UC1);
		Mat sImage = Mat(height, width, CV_8UC1);
		Mat vImage = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b color = src.at<Vec3b>(i, j);

				float normalizedR = (float)color[2] / 255;
				float normalizedG = (float)color[1] / 255;
				float normalizedB = (float)color[0] / 255;

				float M = max(max(normalizedR, normalizedG), normalizedB);
				float m = min(min(normalizedR, normalizedG), normalizedB);
				float C = M - m;


				vImage.at<uchar>(i, j) = M * 255;
				sImage.at<uchar>(i, j) = (M != 0) ? (C / M) * 255 : 0;

				float H = 0;
				if (C != 0) {
					if (M == normalizedR) H = 60 * (normalizedG - normalizedB) / C;
					if (M == normalizedG) H = 120 + 60 * (normalizedB - normalizedR) / C;
					if (M == normalizedB) H = 240 + 60 * (normalizedR - normalizedG) / C;
				}

				if (H < 0) H += 360;

				hImage.at<uchar>(i, j) = H * 255 / 360;
			}

		imshow("Initial Image", src);
		imshow("H Image", hImage);
		imshow("S Image", sImage);
		imshow("V Image", vImage);
		waitKey();
	}
}

void grayscaleHistogram() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int* h = (int*) calloc(256, sizeof(int));

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				h[src.at<uchar>(i, j)] ++;
			}

		imshow("Initial Image", src);
		showHistogram("Histogram", h, 256, height);
		waitKey();
		free(h);
	}
}

std::vector<int> computeMaximums(Mat src) {
	int height = src.rows;
	int width = src.cols;

	int* h = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			h[src.at<uchar>(i, j)] ++;
		}

	float p[256];
	for (int i = 0; i < 256; i++)
		p[i] = (float)h[i] / (height * width);

	free(h);

	int WH = 5;
	float TH = 0.0003;

	std::vector<int> maximums;
	maximums.push_back(0);
	for (int k = 0 + WH; k <= 255 - WH; k++) {
		float avg = 0;
		for (int i = k - WH; i <= k + WH; i++)
			avg += p[i];

		avg /= (2 * WH + 1);

		if (p[k] > avg + TH) {
			float maxim = p[k - WH];
			for (int i = k - WH + 1; i <= k + WH; i++)
				if (maxim < p[i])
					maxim = p[i];
			if (maxim == p[k])
				maximums.push_back(k);
		}
	}
	maximums.push_back(255);
	
	return maximums;
}

int findClosestMax(std::vector<int> maximums, int val) {
	int closestPixel = 0;
	int minDistance = 256;

	for (int maxim : maximums) {
		int dist = abs(val - maxim);
		if (minDistance >= dist) {
			minDistance = dist;
			closestPixel = maxim;
		}
	}

	return closestPixel;
}

void multilevelThresholding() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		std::vector<int> maximums = computeMaximums(src);

		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				int closestPixel = findClosestMax(maximums, src.at<uchar>(i, j));
				dst.at<uchar>(i, j) = closestPixel;
			}

		imshow("Initial Image", src);
		imshow("Threshold Image", dst);
		waitKey();
	}
}

void floydSteinberg() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		std::vector<int> maximums = computeMaximums(src);

		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.at<uchar>(i, j) = src.at<uchar>(i, j);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width ; j++) {
				int oldVal = dst.at<uchar>(i, j);
				int closestPixel = findClosestMax(maximums, oldVal);
				dst.at<uchar>(i, j) = closestPixel;

				int error = oldVal - closestPixel;
				if (j + 1 < width)
					dst.at<uchar>(i, j + 1) = max(min(dst.at<uchar>(i, j + 1) + 7.0 * error / 16, 255), 0);

				if (j > 0 && i + 1 < height)
				dst.at<uchar>(i + 1, j - 1) = max(min(dst.at<uchar>(i + 1, j - 1) + 3.0 * error / 16, 255), 0);

				if (i + 1 < height)
				dst.at<uchar>(i + 1, j ) = max(min(dst.at<uchar>(i + 1, j) + 5.0 * error / 16, 255), 0);

				if (i + 1 < height && j + 1 < width)
				dst.at<uchar>(i + 1, j + 1) = max(min(dst.at<uchar>(i + 1, j + 1) + 1.0 * error / 16, 255), 0);
			}

		imshow("Initial Image", src);
		imshow("Threshold Image", dst);
		waitKey();
	}
}

void createHorizontalProjection(Mat src, Vec3b color) {
	Mat horizontalProjection = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

	int counter;
	for (int i = 0; i < src.rows; i++) {
		counter = 0;
		for (int j = 0; j < src.cols; j++)
			if (src.at<Vec3b>(i, j) == color) {
				horizontalProjection.at<uchar>(i, counter) = 0;
				counter++;
			}
	}

	imshow("Horizontal Projection", horizontalProjection);
}

void createVerticalProjection(Mat src, Vec3b color) {
	Mat verticalProjection = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

	int counter;
	for (int j = 0; j < src.cols; j++) {
		counter = 0;
		for (int i = 0; i < src.rows; i++)
			if (src.at<Vec3b>(i, j) == color) {
				verticalProjection.at<uchar>(counter, j) = 0;
				counter++;
			}
	}

	imshow("verticalProjection", verticalProjection);
}

float computePhiAngle(void* param, int centerOfMass_row, int centerOfMass_column, Vec3b color) {
	Mat* src = (Mat*)param;

	int height = src->rows;
	int width = src->cols;

	int regularRowsCenterSquared = 0;
	int regularColsCenterSquared = 0;
	int rowsAndColsSum = 0;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (src->at<Vec3b>(i, j) == color)
			{
				rowsAndColsSum += (i - centerOfMass_row) * (j - centerOfMass_column);
				regularRowsCenterSquared += (i - centerOfMass_row) * (i - centerOfMass_row);
				regularColsCenterSquared += (j - centerOfMass_column) * (j - centerOfMass_column);
			}

	float phi = atan2(2 * rowsAndColsSum, regularColsCenterSquared - regularRowsCenterSquared) / 2;

	if (phi < 0)
		phi += CV_PI;
	phi = phi * 180 / CV_PI;

	return phi;
}

void geometricFeaturesCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == EVENT_LBUTTONDBLCLK) {
		printf("Pos(x,y): %d, %d  Color(RGB): %d, %d, %d\n",
			x, y,
			(int)src->at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		int height = src->rows;
		int width = src->cols;

		int area = 0;

		int centerOfMass_row = 0;
		int centerOfMass_column = 0;


		int regularRowsCenterSquared = 0;
		int regularColsCenterSquared = 0;
		int rowsAndColsSum = 0;

		Vec3b color = src->at<Vec3b>(y, x);

		float perimeter = 0;
		int dx[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
		int dy[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

		int cMax = -1;
		int cMin = INT_MAX;
		int rMin = INT_MAX;
		int rMax = -1;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src->at<Vec3b>(i, j) == color)
				{
					// compute area
					area++;

					// compute center of mass
					centerOfMass_row += i;
					centerOfMass_column += j;

					// compute perimeter
					bool neighbor = false;
					for (int k = 0; k < 8; k++)
						if (i + dx[k] >= 0 && i + dx[k] <= src->rows && j + dy[k] >= 0 && j + dy[k] <= src->cols)
							if (src->at<Vec3b>(i + dx[k], j + dy[k]) != color)
							{
								neighbor = true;
								break;
							}
					if (neighbor)
						perimeter++;

					// compute aspect ratio
					if (i >= rMax)
						rMax = i;
					if (i <= rMin)
						rMin = i;
					if (j >= cMax)
						cMax = j;
					if (j <= cMin)
						cMin = j;
				}

		printf("Area: %d \n", area);

		centerOfMass_row /= area;
		centerOfMass_column /= area;

		printf("Center of mass: column = %d row = %d \n", centerOfMass_column, centerOfMass_row);

		float phi = computePhiAngle(src, centerOfMass_row, centerOfMass_column, color);
		printf("Phi: %f \n", phi);

		perimeter = perimeter * CV_PI / 4;
		printf("Perimeter = %f \n", perimeter);

		float thinessRatio = 4 * CV_PI * area / (perimeter * perimeter);
		printf("Thiness ratio = %f \n", thinessRatio);

		float aspectRatio = (float)(cMax - cMin + 1) / (rMax - rMin + 1);
		printf("Aspect ratio = %f \n", aspectRatio);

		int rA = centerOfMass_row + tan(phi * CV_PI / 180) * (cMin - centerOfMass_column);
		int rB = centerOfMass_row + tan(phi * CV_PI / 180) * (cMax - centerOfMass_column);

		Point A(cMin, rA);
		Point B(cMax, rB);

		line(*src, A, B, Scalar(0, 0, 0), 2);
		imshow("Elongation Axis", *src);
	
		createHorizontalProjection(*src, color);
		createVerticalProjection(*src, color);
	}
}


void getGeometricFeatures() {
	char fileName[MAX_PATH];
	if (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		//Create a window
		namedWindow("My Window", 1);

		setMouseCallback("My Window", geometricFeaturesCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

Mat_<Vec3b> generateColors(int height, int width, Mat_<int> labels) {
	Mat_<Vec3b> dst(height, width);

	int maxLabel = 0;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (labels(i, j) > maxLabel)
				maxLabel = labels(i, j);

	std::default_random_engine engine;
	std::uniform_int_distribution<int> distribution(0, 255);

	std::vector<Vec3b> colors(maxLabel + 1);

	for (int i = 0; i <= maxLabel; i++) {
		uchar r = distribution(engine);
		uchar g = distribution(engine);
		uchar b = distribution(engine);

		colors.at(i) = Vec3b(r, g, b);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int label = labels(i, j);
			dst(i, j) = (label > 0) ? colors.at(labels(i, j)) : dst(i, j) = Vec3b(255, 255, 255);
		}
	}

	return dst;
}

bool isInside(int height, int width, int i, int j) {
	return i >= 0 && i < height && j >= 0 && j < width;
}

void getBfsLabeling() {
	char fileName[MAX_PATH];
	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat_<int> labels = Mat_<int>(height, width, 0);

		int dx[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
		int dy[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

		int label = 0;
		for (int i = 0; i < height; i++) 
			for (int j = 0; j < width; j++) 
				if (src.at<uchar>(i, j) == 0 && labels(i, j) == 0) {
					label++;
					labels(i, j) = label;

					std::queue<Point> Q;
					Q.push(Point(j, i));

					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++) {
							int nextX = q.x + dx[k];
							int nextY = q.y + dy[k];

							if (isInside(height, width, nextY, nextX)) {
								uchar neighbour = src.at<uchar>(nextY, nextX);

								if (neighbour == 0 && labels(nextY, nextX) == 0) {
									labels(nextY, nextX) = label;
									Q.push(Point(nextX, nextY));
								}
							}
						}
					}
				}

		Mat_<Vec3b> bfs = generateColors(height, width, labels);

		imshow("Initial image", src);
		imshow("BFS", bfs);
		waitKey(0);
	}
}

void getTwoPassLabeling() {
	char fileName[MAX_PATH];
	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat_<int> labels = Mat_<int>(height, width, 0);

		int dx[] = { 0, -1, -1, -1 };
		int dy[] = { -1, -1, 0, 1 };

		int label = 0;
		std::vector<std::vector<int>> edges;
		edges.resize(width * height + 1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src.at<uchar>(i, j) == 0 && labels(i, j) == 0) {
					std::vector<int> L;

					for (int k = 0; k < 4; k++)
					{
						int nextX = i + dx[k];
						int nextY = j + dy[k];
						if (isInside(height, width, nextX, nextY))
							if (labels(nextX, nextY) > 0)
								L.push_back(labels(nextX, nextY));
					}
					
					if (L.size() == 0) {
						label++;
						labels(i, j) = label;
					} else {
						int minElement = *min_element(L.begin(), L.end());

						labels(i, j) = minElement;
						for (int elem : L)
							if (elem != minElement) {
								edges[minElement].push_back(elem);
								edges[elem].push_back(minElement);
							}
					}
				}

		int newLabel = 0;
		int *newLabels = (int *)calloc(width * height + 1, sizeof(int));

		for (int j = 1; j <= label; j++)
			if (newLabels[j] == 0) {
				newLabel++;
				newLabels[j] = newLabel;

				std::queue<int> Q;
				Q.push(j);

				while (!Q.empty()) {
					int poppedElem = Q.front();
					Q.pop();

					for (int elem : edges[poppedElem])
						if (newLabels[elem] == 0) {
							newLabels[elem] = newLabel;
							Q.push(elem);
						}
				}
			}

		for (int i = 0; i < height; i++) 
			for (int j = 0; j < width; j++) 
				labels(i, j) = newLabels[labels(i, j)];


		Mat_<Vec3b> twoPassImg = generateColors(height, width, labels);

		imshow("Initial Image", src);
		imshow("Two-Pass Image", twoPassImg);
		waitKey(0);
	}
}

void borderTracing() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		int dx[] = { 1,  1,  0, -1, -1, -1, 0, 1 };
		int dy[] = { 0, -1, -1, -1,  0,  1, 1, 1 };

		Point P0, P1, Pn, Pn_1;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				if (src.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
					P0 = Point(j, i);
					Pn = Point(j, i);
					i = height;
					j = width;
				}
			}

		int dir = 7;
		int n = 0;

		P1 = Point(P0.x, P0.y);

		std::vector<int> AC, DC;

		do {
			n++;

			Pn_1 = Point(Pn.x, Pn.y);

			if (dir % 2 == 0)
				dir = (dir + 7) % 8;
			else dir = (dir + 6) % 8;

			int j = Pn.x + dx[dir];
			int i = Pn.y + dy[dir];
			Vec3b val = src.at<Vec3b>(i, j);
			while (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
			{
				dir = (dir + 1) % 8;
				j = Pn.x + dx[dir];
				i = Pn.y + dy[dir];
			}
			
			if (P1.x == P0.x && P1.y == P0.y)
				P1 = Point(j, i);

			Pn = Point(j, i);

			AC.push_back(dir);
			src.at<Vec3b>(i, j) = Vec3b(0, 255, 0);

			if (AC.size() >= 2)
			{
				int newDC = (AC.at(AC.size() - 1) - AC.at(AC.size() - 2) + 8) % 8;
				DC.push_back(newDC);
			}
		} while (!((Pn == P1) && (Pn_1 == P0) && (n >= 2)));


		printf("\nAC:");
		for (auto it : AC)
			printf(" %d", it);

		printf("\nDC:");
		for (auto it : DC)
			printf(" %d", it);

		//show the image
		imshow("Image", src);

		waitKey(0);
	}
}

void contourReconstruction() {
	Mat src = imread("./Images/Border_Tracing/gray_background.bmp", IMREAD_GRAYSCALE);

	FILE* fp;
	fp = fopen("./Images/Border_Tracing/reconstruct.txt", "r");

	int x, y;
	fscanf(fp, "%d %d", &y, &x);

	Point P0 = Point(x, y);
	src.at<uchar>(y, x) = 0;

	int n;
	fscanf(fp, "%d", &n);

	int dx[] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	int dy[] = { 0, -1, -1, -1,  0,  1, 1, 1 };

	int dir;
	for (int i = 0; i < n; i++) {
		fscanf(fp, "%d", &dir);

		x = P0.x + dx[dir];
		y = P0.y + dy[dir];

		P0 = Point(x, y);

		src.at<uchar>(y, x) = 0;
	}

	imshow("Image", src);

	waitKey(0);
}

Mat getStructElement() {
	Mat struct_elem(3, 3, CV_8UC1, Scalar(255));

	struct_elem.at<uchar>(0, 1) = 0;
	struct_elem.at<uchar>(1, 1) = 0;
	struct_elem.at<uchar>(1, 0) = 0;
	struct_elem.at<uchar>(1, 2) = 0;
	struct_elem.at<uchar>(2, 1) = 0;

	return struct_elem;
}

Mat dilatationAlg(Mat temp, Mat structuringElem) {
	int height = temp.rows;
	int width = temp.cols;

	Mat dst = temp.clone();

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (temp.at<uchar>(i, j) == 0) {
				for (int si = 0; si < structuringElem.rows; si++)
					for (int sj = 0; sj < structuringElem.cols; sj++)
						if (structuringElem.at<uchar>(si, sj) == 0) {
							int nextI = i + si - structuringElem.rows / 2;
							int nextJ = j + sj - structuringElem.cols / 2;
							if (isInside(height, width, nextI, nextJ)) {
								dst.at<uchar>(nextI, nextJ) = 0;
							}
						}
			}

	return dst;
}

Mat erosionAlg(Mat temp, Mat structuringElem) {
	int height = temp.rows;
	int width = temp.cols;

	Mat dst = temp.clone();

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (temp.at<uchar>(i, j) == 0) {
				for (int si = 0; si < structuringElem.rows; si++)
					for (int sj = 0; sj < structuringElem.cols; sj++)
						if (structuringElem.at<uchar>(si, sj) == 0) {
							int nextI = i + si - structuringElem.rows / 2;
							int nextJ = j + sj - structuringElem.cols / 2;
							if (isInside(height, width, nextI, nextJ)) {
								if (temp.at<uchar>(nextI, nextJ) == 255)
									dst.at<uchar>(i, j) = 255;
							}
						}
			}

	return dst;
}

void getDilatation(int n) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;
		
		Mat dst = src.clone();

		while (n > 0) {
			Mat temp = dilatationAlg(dst, structuringElem);
			dst = temp.clone();
			n--;
		}

		imshow("Initial Image", src);
		imshow("Dilatation Image", dst);

		waitKey(0);
	}
}

void getErosion(int n) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;

		Mat dst = src.clone();

		while (n > 0) {
			Mat temp = erosionAlg(dst, structuringElem);
			dst = temp.clone();
			n--;
		}

		imshow("Initial Image", src);
		imshow("Erosion Image", dst);

		waitKey(0);
	}
}

void getOpening() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;

		Mat temp = erosionAlg(src, structuringElem);

		Mat dst = dilatationAlg(temp, structuringElem);

		imshow("Initial Image", src);
		imshow("Opening Image", dst);

		waitKey(0);
	}
}

void getClosing() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;

		Mat temp = dilatationAlg(src, structuringElem);

		Mat dst = erosionAlg(temp, structuringElem);

		imshow("Initial Image", src);
		imshow("Closing Image", dst);

		waitKey(0);
	}
}

void getBoundaryExtraction() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;

		Mat dst = src.clone();

		Mat temp = erosionAlg(src, structuringElem);

		for (int i = 0; i < height; i++) 
			for (int j = 0; j < width; j++) 
				dst.at<uchar>(i, j) = (src.at<uchar>(i, j) == temp.at<uchar>(i, j)) ? 255 : 0;
	
		imshow("Initial Image", src);
		imshow("Boundary Extraction Image", dst);

		waitKey(0);
	}
}

bool areEqual(Mat mat1, Mat mat2) {
	for (int i = 0; i < mat1.rows; i++)
		for (int j = 0; j < mat1.cols; j++)
			if (mat1.at<uchar>(i, j) != mat2.at<uchar>(i, j))
				return false;
	
	return true;
}

void getRegionFilling() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		Mat structuringElem = getStructElement();

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		dst.at<uchar>(height / 2, width / 2) = 0;

		Mat temp = Mat(height, width, CV_8UC1, Scalar(255));

		Mat neg = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				neg.at<uchar>(i, j) = (src.at<uchar>(i, j) == 0) ? 255 : 0;

		while (1) {
			Mat dilatation = dilatationAlg(dst, structuringElem);

			for (int i = 0; i < height; i++) 
				for (int j = 0; j < width; j++) 
					if (dilatation.at<uchar>(i, j) == 0 && neg.at<uchar>(i, j) == 0)
						temp.at<uchar>(i, j) = 0;
					else temp.at<uchar>(i, j) = 255;

			if (areEqual(dst, temp))
				break;
			
			dst = temp.clone();
		}

		imshow("Initial Image", src);
		imshow("Region Filling Image", dst);

		waitKey(0);
	}
}

void computeHistogram(Mat img, int* histogram) {
	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			histogram[img.at<uchar>(i, j)]++;
}

float getMeanValue(int height, int width, int histogram[256]) {
	int M = height * width;

	float g = 0;
	for (int i = 0; i < 256; i++)
		g += i * histogram[i];

	return (float)g / M;
}

float getStandardDeviation(int height, int width, int histogram[256], float meanValue) {
	int M = height * width;

	float deviation = 0;
	for (int i = 0; i < 256; i++)
		deviation += (i - meanValue) * (i - meanValue) * histogram[i];

	deviation = (float)deviation / M;
	return sqrt(deviation);
}

void getGlobalThresholding(Mat img, int histogram[256]) {
	int I_MIN = 256;
	int I_MAX =  -1;
	
	for (int i = 0; i < 256; i++) {
		if (histogram[i] != 0 && I_MIN > i)
			I_MIN = i;
		if (histogram[i] != 0 && I_MAX < i)
			I_MAX = i;
	}

	float current_T = (float)(I_MIN + I_MAX) / 2;
	float previous_T = current_T;

	do {
		float meanValue1 = 0;
		float N1 = 0;
		for (int i = I_MIN; i < (int) current_T; i++) {
			meanValue1 += i * histogram[i];
			N1 += histogram[i];
		}
		meanValue1 /= N1;

		float meanValue2 = 0;
		float N2 = 0;
		for (int i = (int) current_T; i < I_MAX; i++) {
			meanValue2 += i * histogram[i];
			N2 += histogram[i];
		}
		meanValue2 /= N2;

		previous_T = current_T;
		current_T = (float)(meanValue1 + meanValue2) / 2;
	} while (abs(current_T - previous_T) > 0.1);

	int height = img.rows;
	int width = img.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = (img.at<uchar>(i, j) < current_T) ? 0 : 255;
		}
	std::cout << "Threshold : " << current_T << "\n";
	imshow("Black & White Image", dst);
}

void getStatistics() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		int* histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(src, histogram);
		showHistogram("Histogram", histogram, 256, height);

		float meanValue = getMeanValue(height, width, histogram);
		std::cout << "Mean Value : " << meanValue << "\n";

		float deviation = getStandardDeviation(height, width, histogram, meanValue);
		std::cout << "Standard deviation : " << deviation << "\n";

		getGlobalThresholding(src, histogram);
		
		imshow("Initial Image", src);

		free(histogram);
		waitKey(0);
	}
}

void brightnessChange(int offset) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int* histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(src, histogram);
		showHistogram("Initial Histogram", histogram, 256, height);;
		imshow("Initial Image", src);
		free(histogram);

		for (int i = 0; i < height; i++) 
			for (int j = 0; j < width; j++) {
				uchar value = src.at<uchar>(i, j) + offset;

				if (value < 0)
					value = 0;
				if (value > 255)
					value = 255;

				dst.at<uchar>(i, j) = value;
			}

		histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(dst, histogram);
		showHistogram("New Histogram", histogram, 256, height);;
		imshow("New Image", dst);

		free(histogram);
		waitKey(0);
	}
}

void contrastChange(int min, int max) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int* histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(src, histogram);
		showHistogram("Initial Histogram", histogram, 256, height);;
		imshow("Initial Image", src);

		int I_MIN = 256;
		int I_MAX = -1;

		for (int i = 0; i < 256; i++) {
			if (histogram[i] != 0 && I_MIN > i)
				I_MIN = i;
			if (histogram[i] != 0 && I_MAX < i)
				I_MAX = i;
		}
		free(histogram);

		float raport = (float)(max - min) / (I_MAX - I_MIN);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar value = min + (src.at<uchar>(i, j) - I_MIN) * raport;

				if (value < 0)
					value = 0;
				if (value > 255)
					value = 255;

				dst.at<uchar>(i, j) = value;
			}

		histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(dst, histogram);
		showHistogram("New Histogram", histogram, 256, height);;
		imshow("New Image", dst);

		free(histogram);
		waitKey(0);
	}
}

void gammaCorrection(float gamma) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int* histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(src, histogram);
		showHistogram("Initial Histogram", histogram, 256, height);;
		imshow("Initial Image", src);
		free(histogram);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar value = 255 * (float) pow((float) src.at<uchar>(i, j) / 255, gamma);

				if (value < 0)
					value = 0;
				if (value > 255)
					value = 255;

				dst.at<uchar>(i, j) = value;
			}

		histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(dst, histogram);
		showHistogram("New Histogram", histogram, 256, height);;
		imshow("New Image", dst);

		free(histogram);
		waitKey(0);
	}
}

void histogramEqualization() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int* histogram = (int *)calloc(256, sizeof(int));
		computeHistogram(src, histogram);
		showHistogram("Initial Histogram", histogram, 256, height);;
		imshow("Initial Image", src);

		int M = height * width;
		float* cpdf = (float*)malloc(256 * sizeof(float));

		cpdf[0] = (float) histogram[0] / M;
		for (int i = 1; i < 256; i++)
			cpdf[i] = cpdf[i - 1] + (float) histogram[i] / M;
		free(histogram);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar value = 255 * cpdf[src.at<uchar>(i, j)];

				if (value < 0)
					value = 0;
				if (value > 255)
					value = 255;

				dst.at<uchar>(i, j) = value;
			}

		histogram = (int*)calloc(256, sizeof(int));
		computeHistogram(dst, histogram);
		showHistogram("New Histogram", histogram, 256, height);;
		imshow("New Image", dst);

		free(histogram);
		waitKey(0);
	}
}

Mat_<float> getConvolution(Mat_<uchar> img, Mat_<float> H) {
	int height = img.rows;
	int width = img.cols;

	int filterSize_rows = H.rows;
	int filterSize_cols = H.cols;

	Mat_<float> convolution(height, width);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			float sum = 0;

			for (int ii = 0; ii < filterSize_rows; ii++)
				for (int jj = 0; jj < filterSize_cols; jj++) {
					int neighbour_i = i + ii - filterSize_rows / 2;
					int neighbour_j = j + jj - filterSize_cols / 2;
					if (isInside(height, width, neighbour_i, neighbour_j)) 
						sum += H(ii, jj) * img(neighbour_i, neighbour_j);
				}

			convolution(i, j) = sum;
		}

	return convolution;
}

Mat_<uchar> normalizeConvolution(Mat_<float> H, Mat_<float> convolution, int normalCoeff) {
	int height = convolution.rows;
	int width = convolution.cols;

	int filterRows = H.rows;

	Mat_<uchar> dst(height, width);

	float sumNegative = 0.0;
	float sumPositive = 0.0;

	bool lowPassFilter = true;

	for (int i = 0; i < filterRows; i++)
		for (int j = 0; j < filterRows; j++)
			if (H(i, j) > 0) 
				sumPositive += H(i, j);
			else if (H(i, j) < 0) {
				sumNegative += abs(H(i, j));
				lowPassFilter = false;
			}

	if (lowPassFilter) { // LOW PASS
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) 
				dst(i, j) = convolution(i, j) / (normalCoeff * 1.0f);
	} else {	// HIGH FILTER
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (convolution(i, j) < 0)
					dst(i, j) = 0;
				else if (convolution(i, j) > 255)
					dst(i, j) = 255;
				else dst(i, j) = convolution(i, j);
	}

	return dst;
}

void spatialDomainFilter() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat_<uchar> src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		imshow("Initial Image", src);

		// mean filter 3x3
		Mat_<float> H1(3, 3, (float)1);
		Mat_<float> convolution = getConvolution(src, H1);
		Mat_<uchar> dst = normalizeConvolution(H1, convolution, 9);
		imshow("Mean Filter 3x3 Image", dst);

		// mean filter 5x5
		Mat_<float> H2(5, 5, (float)1);
		convolution = getConvolution(src, H2);
		dst = normalizeConvolution(H2, convolution, 25);
		imshow("Mean Filter 5x5 Image", dst);

		// gaussian filter 3x3
		Mat_<float> H3(3, 3, (float)1);
		H3(0, 1) = 2;
		H3(1, 0) = 2;
		H3(1, 2) = 2;
		H3(2, 1) = 2;
		H3(1, 1) = 4;

		convolution = getConvolution(src, H3);
		dst = normalizeConvolution(H3, convolution, 16);
		imshow("Gaussian Filter 5x5 Image", dst);

		// laplace filter 3x3
		Mat_<float> H4(3, 3, (float)0);
		H4(0, 1) = -1;
		H4(1, 0) = -1;
		H4(1, 2) = -1;
		H4(2, 1) = -1;
		H4(1, 1) = 4;

		convolution = getConvolution(src, H4);
		dst = normalizeConvolution(H4, convolution, 1);
		imshow("Laplace Filter 5x5 Image", dst);

		// high-pass filter 3x3
		Mat_<float> H5(3, 3, (float)0);
		H5(0, 1) = -1;
		H5(1, 0) = -1;
		H5(1, 2) = -1;
		H5(2, 1) = -1;
		H5(1, 1) = 5;

		convolution = getConvolution(src, H5);
		dst = normalizeConvolution(H5, convolution, 1);
		imshow("High-Pass Filter 5x5 Image", dst);

		waitKey(0);
	}
}

void centerImage(Mat img) {
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
}

void frequencyDomainFilter() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat src_f;
		src.convertTo(src_f, CV_32FC1);

		centerImage(src_f);

		Mat fourier;
		dft(src_f, fourier, DFT_COMPLEX_OUTPUT);

		Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		split(fourier, channels);

		Mat mag, phi;
		magnitude(channels[0], channels[1], mag);
		phase(channels[0], channels[1], phi);

		float maxLog = 0.0;
		for (int i = 0; i < mag.rows; i++)
			for (int j = 0; j < mag.cols; j++) {
				int v = log(mag.at<float>(i, j) + 1); 
				maxLog = (v > maxLog) ? v : maxLog;
			}

		Mat normalizedMagnitude(mag.rows, mag.cols, CV_8UC1);
		for (int i = 0; i < mag.rows; i++)
			for (int j = 0; j < mag.cols; j++) {
				float v = log(mag.at<float>(i, j) + 1);
				normalizedMagnitude.at<uchar>(i, j) = v * 255 / maxLog;
			}

		imshow("Initial Image", src);
		imshow("Magnitude Image", normalizedMagnitude);

		int H = fourier.rows;
		int W = fourier.cols;
		Mat idealLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		Mat idealHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		for (int i = 0; i < H; i++)
			for (int j = 0; j < W; j++) {
				float check = (i - H / 2) * (i - H / 2) + (j - W / 2) * (j - W / 2);
				if (check > 100) { // R = 10 
					idealLowChannels[0].at<float>(i, j) = 0;
					idealLowChannels[1].at<float>(i, j) = 0;

					idealHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					idealHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);
				}
				else {
					idealLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					idealLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);

					idealHighChannels[0].at<float>(i, j) = 0;
					idealHighChannels[1].at<float>(i, j) = 0;
				}
			}

		Mat idealLowDst, idealLowDst_f;
		merge(idealLowChannels, 2, fourier);
		dft(fourier, idealLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centerImage(idealLowDst_f);
		normalize(idealLowDst_f, idealLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("Ideal Low Filter Image", idealLowDst);

		Mat idealHighDst, idealHighDst_f;
		merge(idealHighChannels, 2, fourier);
		dft(fourier, idealHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centerImage(idealHighDst_f);
		normalize(idealHighDst_f, idealHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("Ideal High Filter Image", idealHighDst);

		Mat gaussLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		Mat gaussHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		for (int i = 0; i < H; i++)
			for (int j = 0; j < W; j++) {
				gaussLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);
				gaussLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);
			
				gaussHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
				gaussHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
			}

		Mat gaussLowDst, gaussLowDst_f;
		merge(gaussLowChannels, 2, fourier);
		dft(fourier, gaussLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centerImage(gaussLowDst_f);
		normalize(gaussLowDst_f, gaussLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("Gaussian-Cut Low Filter Image", gaussLowDst);

		Mat gaussHighDst, gaussHighDst_f;
		merge(gaussHighChannels, 2, fourier);
		dft(fourier, gaussHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		centerImage(gaussHighDst_f);
		normalize(gaussHighDst_f, gaussHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
		imshow("Gaussian-Cut High Filter Image", gaussHighDst);

		waitKey(0);
	}
}

//Lab 10
void saltPeper(int w) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1);
		std::vector<uchar> v;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				v.clear();

				for (int k = 0; k < w; k++)
					for (int l = 0; l < w; l++) {
						int nextI = i - (w / 2) + k;
						int nextJ = j - (w / 2) + l;

						if (isInside(height, width, nextI, nextJ))
							v.push_back(src.at<uchar>(nextI, nextJ));
					}

				std::sort(v.begin(), v.end());
				dst.at<uchar>(i, j) = v.at(v.size() / 2);
			}

		imshow("Initial Image", src);
		imshow("SaltPepper", dst);

		waitKey(0);
	}
}

void gaussian_1(float sigma) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1);

		float aux = 6 * sigma;
		int w = aux + 0.5;
		if (w % 2 == 0) 
			w++;

		Mat gauss(w, w, CV_32F);
		float sum = 0;
		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++) {
				gauss.at<float>(i, j) = ((1 / (2 * CV_PI * sigma * sigma)) * exp(-(pow((i - (w / 2)), 2) + pow((j - (w / 2)), 2)) / (2 * sigma * sigma)));
				sum += gauss.at<float>(i, j);
			}

		printf("Gauss 1 Sum: %f\n", sum);

		double t = (double)getTickCount();

		Mat convolution = getConvolution(src, gauss);

		t = ((double)getTickCount() - t) / getTickFrequency();

		printf("Time: %.3f [ms]\n", t * 1000);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.at<uchar>(i, j) = convolution.at<float>(i, j) / sum;

		imshow("Initial Image", src);
		imshow("Gauss 1", dst);

		waitKey(0);
	}
}

Mat gaussian_noise_filter(Mat src, float sigma) {
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_8UC1);

	float aux = 6 * sigma;
	int w = aux + 0.5;
	if (w % 2 == 0)
		w++;

	Mat gauss(w, w, CV_32F);
	float sum = 0;
	for (int i = 0; i < w; i++)
		for (int j = 0; j < w; j++) {
			gauss.at<float>(i, j) = ((1 / (2 * CV_PI * sigma * sigma)) * exp(-(pow((i - (w / 2)), 2) + pow((j - (w / 2)), 2)) / (2 * sigma * sigma)));
			sum += gauss.at<float>(i, j);
		}

	printf("Gauss 2 Sum: %f\n", sum);

	Mat gauss_1d = Mat(w, 1, CV_32F);
	Mat gauss_2d = Mat(1, w, CV_32F);
	float sum_1d = 0;
	float sum_2d = 0;
	for (int i = 0; i < w; i++) {
		gauss_1d.at<float>(i, 0) = gauss.at<float>(i, (w / 2));
		sum_1d += gauss_1d.at<float>(i, 0);
	}

	for (int j = 0; j < w; j++) {
		gauss_2d.at<float>(0, j) = gauss.at<float>((w / 2), j);
		sum_2d += gauss_2d.at<float>(0, j);
	}

	double t = (double)getTickCount();
	Mat conv = getConvolution(src, gauss_1d);
	t = ((double)getTickCount() - t) / getTickFrequency();

	Mat tmp = Mat(height, width, CV_32F);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			tmp.at<float>(i, j) = conv.at<float>(i, j) / sum_1d;

	double t2 = (double)getTickCount();
	Mat conv2 = getConvolution(tmp, gauss_2d);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			dst.at<uchar>(i, j) = conv2.at<float>(i, j) / sum_2d;


	printf("Time: %.3f [ms]\n", (t + t2) * 1000);
	
	return dst;
}

void gaussian_2(float sigma) {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		Mat dst = gaussian_noise_filter(src, sigma);

		imshow("Initial Image", src);
		imshow("Gauss 2", dst);

		waitKey(0);
	}
}

Mat_<uchar> non_max_suppression(Mat_<float> gradient, Mat_<float> theta) {
	int height = gradient.rows;
	int width = gradient.cols;

	Mat_<uchar> result(height, width);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			float angle = theta(i, j) < 0 ? theta(i, j) + CV_PI : theta(i, j);
			float q = 0, r = 0;

			int region;
			if ((angle <= CV_PI / 8.0 || angle >= 7 * CV_PI / 8.0) 
				&& isInside(height, width, i, j + 1) && isInside(height, width, i, j - 1)) {
				q = gradient(i, j + 1);
				r = gradient(i, j - 1);
			}
			else if ((angle >= CV_PI / 8.0 && angle <= 3 * CV_PI / 8.0)
				&& isInside(height, width, i - 1, j + 1) && isInside(height, width, i + 1, j - 1)) {
				q = gradient(i + 1, j - 1);
				r = gradient(i - 1, j + 1);
			}
			else if ((angle >= 3 * CV_PI / 8.0 && angle <= 5 * CV_PI / 8.0)
				&& isInside(height, width, i - 1, j) && isInside(height, width, i + 1, j)){
				q = gradient(i + 1, j);
				r = gradient(i - 1, j);
			}
			else if(isInside(height, width, i - 1, j - 1) && isInside(height, width, i + 1, j + 1)) {
				q = gradient(i - 1, j - 1);
				r = gradient(i + 1, j + 1);
			}

			if ((gradient(i, j) >= q) && (gradient(i, j) >= r))
				result(i, j) = gradient(i, j);
			else
				result(i, j) = 0;
		}

	return result;
}

Mat_<uchar> adaptiveThresholding(Mat_<uchar> src) {
	int height = src.rows;
	int width = src.cols;

	int* h = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			h[src(i, j)]++;
		}
	showHistogram("Histogram", h, 256, height);

	int noNonZeroEdges = 0.1 * (height * width - h[0]);

	int sum = h[255];
	int tHigh = 255;
	while (sum < noNonZeroEdges) {
		tHigh--;
		sum += h[tHigh];
	}
	free(h);

	int tLow = 0.4 * tHigh;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			if (src(i, j) >= tHigh)
				src(i, j) = 255;
			else if (src(i, j) >= tLow)
				src(i, j) = 127;
			else src(i, j) = 0;
		}

	int dx[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dy[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (src(i, j) == 255) {
				std::queue<Point> Q;
				Q.push(Point(j, i));

				while (!Q.empty()) {
					Point q = Q.front();
					Q.pop();

					for (int k = 0; k < 8; k++) {
						int nextI = q.y + dy[k];
						int nextJ = q.x + dx[k];

						if (isInside(height, width, nextI, nextJ)) {
							if (src(nextI, nextJ) == 127) {
								src(nextI, nextJ) = 255;
								Q.push(Point(nextJ, nextI));
							}
						}
					}
				}
			}
	
	return src;
}

Mat_<uchar> getEdgeLinking(Mat_<uchar> src) {
	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (src(i, j) == 127)
				src(i, j) = 0;

	return src;
}


void cannyEdgeDetection() {
	char fileName[MAX_PATH];

	while (openFileDlg(fileName)) {
		Mat src = imread(fileName, IMREAD_GRAYSCALE);
		imshow("Initial Image", src);

		int height = src.rows;
		int width = src.cols;

		Mat filtered_image = gaussian_noise_filter(src, 0.5);
		imshow("Filtered Image", filtered_image);
		
		Mat_<float> Sx(3, 3);
		Sx(0, 0) = -1;
		Sx(0, 1) = 0;
		Sx(0, 2) = 1;
		Sx(1, 0) = -2;
		Sx(1, 1) = 0;
		Sx(1, 2) = 2;
		Sx(2, 0) = -1;
		Sx(2, 1) = 0;
		Sx(2, 2) = 1;

		Mat_<float> Sy(3, 3);
		Sy(0, 0) = 1;
		Sy(0, 1) = 2;
		Sy(0, 2) = 1;
		Sy(1, 0) = 0;
		Sy(1, 1) = 0;
		Sy(1, 2) = 0;
		Sy(2, 0) = -1;
		Sy(2, 1) = -2;
		Sy(2, 2) = -1;

		Mat_<float> convolutionX = getConvolution(src, Sx);
		Mat_<float> convolutionY = getConvolution(src, Sy);
		Mat_<float> gradient(convolutionX.rows, convolutionX.cols);
		Mat_<float> theta(convolutionX.rows, convolutionX.cols);

		for (int i = 0; i < convolutionX.rows; i++)
			for (int j = 0; j < convolutionX.cols; j++) {
				gradient(i, j) = sqrt(pow(convolutionX(i, j), 2) + pow(convolutionY(i, j), 2));
				theta(i, j) = atan2(convolutionY(i, j), convolutionX(i, j));
			}

		Mat_<uchar> normalizedGradient(convolutionX.rows, convolutionX.cols);
		for (int i = 0; i < convolutionX.rows; i++)
			for (int j = 0; j < convolutionX.cols; j++) {
				gradient(i, j) = gradient(i, j) / (4 * sqrt(2));
				normalizedGradient(i, j) = gradient(i, j);
			}

		imshow("Gradient", normalizedGradient);

		Mat_<uchar> non_max = non_max_suppression(gradient, theta);
		imshow("Non Max Suppression", non_max);

		Mat_<uchar> threshold = adaptiveThresholding(non_max);
		imshow("Adaptive Thresholding", threshold);

		Mat_<uchar> edgeLinking = getEdgeLinking(threshold);
		imshow("Edge Linking", edgeLinking);

		waitKey(0);
	}
}

int main() {
	int op;
	do {
		system("cls");

		destroyAllWindows();

		printf("Menu:\n");
		printf(" 1  - Open image\n");
		printf(" 2  - Open BMP images from folder\n");
		printf(" 3  - Image negative - diblook style\n");
		printf(" 4  - BGR -> HSV\n"); // lab 2
		printf(" 5  - Resize image\n");
		printf(" 6  - Canny edge detection\n");
		printf(" 7  - Edges in a video sequence\n");
		printf(" 8  - Snap frame from live video\n");
		printf(" 9  - Mouse callback demo\n");
		//lab 1
		printf(" Lab 1\n");
		printf(" 10 - Change Gray Level -> Lighten\n");
		printf(" 11 - Create 4 Colors Image\n");
		printf(" 12 - Inverse a float matrix\n");
		//lab 2
		printf(" Lab 2\n");
		printf(" 13 - Split color image\n");
		printf(" 14 - RGB -> GrayScale\n");
		printf(" 15 - GrayScale to Black & White\n");
		//lab 3
		printf(" Lab 3\n");
		printf(" 16 - GrayScale Histogram\n");
		printf(" 17 - Multilevel Thresholding\n");
		printf(" 18 - Floyd Steinberg\n");
		//lab 4
		printf(" Lab 4\n");
		printf(" 19 - Geometric Features\n");
		// lab 5
		printf(" Lab 5\n");
		printf(" 20 - Labeling: BFS\n");
		printf(" 21 - labeling: Two-pass Traversal\n");
		// lab 6
		printf(" Lab 6\n");
		printf(" 22 - Border Tracing\n");
		printf(" 23 - Contour Reconstruction\n");
		// lab 7
		printf(" Lab 7\n");
		printf(" 24 - Dilatation\n");
		printf(" 25 - Erosion\n");
		printf(" 26 - Opening\n");
		printf(" 27 - Closing\n");
		printf(" 28 - Boundary Extraction\n");
		printf(" 29 - Region Filling\n");
		// lab 8
		printf(" Lab 8\n");
		printf(" 30 - Statistical Characteristics\n");
		printf(" 31 - Brightness Change\n");
		printf(" 32 - Contrast Change\n");
		printf(" 33 - Gamma Correction\n");
		printf(" 34 - Histogram Equalization\n");
		// lab 9
		printf(" Lab 9\n");
		printf(" 35 - Spatial Domain Filter\n");
		printf(" 36 - Frequency Domain Filter\n");
		// lab 10
		printf(" Lab 10\n");
		printf(" 37 - Salt & Pepper\n");
		printf(" 38 - Gauss 1\n");
		printf(" 39 - Gauss 2\n");
		// lab 10
		printf(" Lab 11\n");
		printf(" 40 - Canny\n");
		printf("  0  - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		int n, m;
		float g;

		switch (op) {
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				rgbToHSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				changeGrayScale();
				break;
			case 11:
				createFourColorsImage();
				break;
			case 12:
				inverseFloatMatrix();
				break;
			case 13:
				splitImage();
				break;
			case 14:
				rgbToGrayscale();
				break;
			case 15:
				grayscaleToBW();
				break;
			case 16:
				grayscaleHistogram();
				break;
			case 17:
				multilevelThresholding();
				break;
			case 18:
				floydSteinberg();
				break;
			case 19:
				getGeometricFeatures();
				break;
			case 20:
				getBfsLabeling();
				break;
			case 21:
				getTwoPassLabeling();
				break;
			case 22:
				borderTracing();
				break;
			case 23:
				contourReconstruction();
				break;
			case 24:
				scanf("%d", &n);
				getDilatation(n);
				break;
			case 25:
				scanf("%d", &n);
				getErosion(n);
				break;
			case 26:
				getOpening();
				break;
			case 27:
				getClosing();
				break;
			case 28:
				getBoundaryExtraction();
				break;
			case 29:
				getRegionFilling();
				break;
			case 30:
				getStatistics();
				break;
			case 31:
				scanf("%d", &n);
				brightnessChange(n);
				break;
			case 32:
				scanf("%d %d", &n, &m);
				contrastChange(n, m);
				break;
			case 33:
				scanf("%f", &g);
				gammaCorrection(g);
				break;
			case 34:
				histogramEqualization();
				break;
			case 35:
				spatialDomainFilter();
				break;
			case 36:
				frequencyDomainFilter();
				break;
			case 37:
				scanf("%d", &n);
				saltPeper(n);
				break;
			case 38:
				scanf("%f", &g);
				gaussian_1(g);
				break;
			case 39:
				scanf("%f", &g);
				gaussian_2(g);
				break;
			case 40:
				cannyEdgeDetection();
				break;
		}
	} while (op != 0);

	return 0;
}