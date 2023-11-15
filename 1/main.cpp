#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_world440d.lib") #else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_world440.lib")
#endif

void proc(const Mat& img, Mat& out){
    int w = img.cols, h = img.rows;

    img.copyTo(out);
    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            Vec3b& p = out.at<Vec3b>(y, x);
            p = p * 0.5;
            p[1] = 0;
        }
    }
    for(int i = 0; i < 100; i++)
        circle(out, Point(rand()%w, rand()%h), 10, CV_RGB(rand()%255, rand() % 255, rand() % 255), -1, LINE_AA);

    rectangle(out, Rect(150, 100, 200, 100), CV_RGB(255, 0, 0), -1, LINE_AA);

    for (int x = 0; x < w; x += 100)
        line(out, Point(x, 0), Point(x, h), CV_RGB(0,255,0), 1, LINE_AA);
    for (int y = 0; y < h; y += 100)
        line(out, Point(0, y), Point(w, y), CV_RGB(0, 255, 0), 1, LINE_AA);
    putText(out, "Test", Point(30, 130), FONT_HERSHEY_SCRIPT_SIMPLEX, 3.0, CV_RGB(255,255,0), 3.0, LINE_AA);
}
void ichi(const Mat& img, Mat& out){
    int w = img.cols, h = img.rows;

    img.copyTo(out);
    float sw2, sh2;
    sw2 = w/10;
    sh2 = h/10;

    for (int i = 1; i<11; i++){
        float x = 0;
        if (i % 2 != 0){
            x = sh2;
        }
        for (int j = 1; j < 6; j++){
            rectangle(out, Point(sw2 * (i - 1), x), Point(sw2 * i, x + sh2), Scalar(255, 0, 0), -1);
            x += 2 * sh2;
        }
    }
    putText(out, "234-D8719", Point(30, 130), FONT_HERSHEY_SCRIPT_SIMPLEX, 3.0, CV_RGB(255, 255, 0), 3.0, LINE_AA);
}
int main() {
    Mat img = imread("in.jpg");

    int w = img.cols, h = img.rows;
    int sw = 800, sh = 800 * h / w;
    resize(img, img, Size(sw, sh));

    Mat out;
    ichi(img, out);
    imshow("out", out);
    imwrite("out.jpg", out);
    waitKey(0);
}