#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;


#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

void sobelFilter(const Mat& input, Mat& output) {
    CV_Assert(input.channels() == 1);


    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

  
    output = Mat::zeros(input.size(), CV_8U);

    
#pragma omp parallel for collapse(2)
    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += Gx[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                    sumY += Gy[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                }
            }

            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            magnitude = magnitude > 255 ? 255 : magnitude;
            magnitude = magnitude < 0 ? 0 : magnitude;

            output.at<uchar>(y, x) = magnitude;
        }
    }
}

int main() {
    
    string imagePath = "C:/Users/老于/Desktop/OIP.jpg";
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (!image.data) {
        cout << "No image data" << endl;
        return -1;
    }

  
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    
    Mat sobelImage;

   
    double start = omp_get_wtime();
    sobelFilter(grayImage, sobelImage);
    double end = omp_get_wtime();
    cout << "Parallel Sobel filter applied in " << (end - start) << " seconds." << endl;

    
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    namedWindow("Sobel Image", WINDOW_AUTOSIZE);
    imshow("Sobel Image", sobelImage);

    waitKey(0);

    return 0;
}

