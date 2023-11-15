#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(){
    Mat img, img_gray, img_canny, img_bila, img_comp;
    img = imread("./in.jpg");
    resize(img, img, Size(800, 800 * img.rows / img.cols));

    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    Canny(img_gray, img_canny, 30, 80); // 入力画像，出力画像，閾値1，閾値2
    bitwise_not(img_canny, img_canny); //　色反転
    cvtColor(img_canny, img_canny,cv::COLOR_GRAY2BGR);
    
    bilateralFilter(img, img_bila, -1, 70, 75);
    
    bitwise_and(img_bila, img_canny, img_comp);

    imwrite("./out_canny.png", img_canny);
    imwrite("./out_bila.png", img_bila);
    imwrite("./out_comp.png", img_comp);
}