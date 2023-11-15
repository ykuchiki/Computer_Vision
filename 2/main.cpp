#include <stdio.h>
#include <sstream> 
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

// [-1, +1]のガウシアンノイズの生成
float Box_Muller() { 
    float X, Y;
    X = (rand() % 10000) / 10000.0;  // 0~9999の乱数を10000.0で正規化している
    Y = (rand() % 10000) / 10000.0;

    return sqrt(-2 * log(X)) * cos(2 * 3.14159 * Y);  // Box_Mullerは2つの別の乱数を使って計算する．sinのパターンも使える
}

// 画像にノイズを加える
void add_noise(Mat& img, float sigma, Mat& out){
    img.copyTo(out);
    for (int ptr=0; ptr < 3 * out.cols * out.rows; ptr++){
        int v = out.data[ptr] + Box_Muller() * sigma;
        if (v < 0)v = 0;
        if (v > 255)v  = 255;
        out.data[ptr] = v;
    }
} 

// フィルタかける関数
void apply_filter(Mat& in, Mat& out, Mat& f){
    int w, h, fw, fh;
    w = in.cols; h = in.rows;
    fw = f.cols; fh = f.rows;

    if (fw != fh || fw % 2 == 0)return;  // フィルタが奇数の正方形であること(中心を持つ)

    int N = (fw - 1) / 2;
    out = Mat(h, w, CV_8UC3);
    for (int y = 0; y < h ; y++){
        for (int x = 0; x < w ; x++){

            Vec3b p = 0;

            for (int sy = -N; sy <= N; sy++){
                // 境界処理
                int ty = y + sy;
                if (ty < 0)ty = -ty; else if (ty >= h) ty = h - 1 - (h - sy);
                for (int sx = -N; sx <= N; sx++){
                    int tx = x + sx;
                    if (tx < 0)tx = -tx; else if (tx >= w) tx = w - 1 - (w -sx);
                    p = p + in.at<Vec3b>(ty, tx) * f.at<double>(sy + N, sx + N);
                } //sx
            }//sy

            out.at<Vec3b>(y,x) = p;
        }//x
    }//y

}

// ガウシアンフィルタ
Mat gausian(int N){
    float sigma = N / 6.0;
    // int N = floor(sigma * 3.0) * 2 + 1;
    Mat h(N, N, CV_64F);

    double sum = 0;
    for (int sy = 0; sy < N; sy++){
        for (int sx = 0; sx < N; sx++){
            int x = sx - (N - 1) / 2;
            int y = sy - (N - 1) / 2;
            double g = exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2 * 3.14159 * sigma * sigma);
            h.at<double>(sy, sx) = g;
            sum += g;
        }
    }
    return h = h * (1 / sum);
}

// PSNR
double psnr(Mat& i1, Mat& i2){
    Mat s1;
    absdiff(i1, i2, s1);  // |i1 -i2|
    s1.convertTo(s1, CV_64F);
    s1 = s1.mul(s1);  //s1**2

    Scalar s =sum(s1);
    double sse1 = s.val[0];
    double sse2 = s.val[1];
    double sse3 = s.val[2];
    double rmse1 = sqrt(sse1 / double(i1.total()));
    double rmse2 = sqrt(sse2 / double(i1.total()));
    double rmse3 = sqrt(sse3 / double(i1.total()));

    double psnr1 = 10.0 * log10((255 * 255) / rmse1);
    double psnr2 = 10.0 * log10((255 * 255) / rmse2);
    double psnr3 = 10.0 * log10((255 * 255) / rmse3);

    double psnr = (psnr1 + psnr2 + psnr3) / 3;
    return psnr;
}

void psnr_text(Mat& img, Mat& img2, String& filename){
    // Mat img = imread("in.jpg");
    // const char *mode = "gausian";
    // const float fnum = 15.0;
    // std::ostringstream filenameStream;
    // filenameStream << "out_" << fnum << mode <<".png";

    // std::string filename = filenameStream.str();
    // cv::Mat img2 = cv::imread(filename);
    double s = psnr(img, img2);

    std::ofstream outFile("psnr_results.txt", std::ios::app);
    outFile << filename << "のpsnr" << s;
    outFile << "\n";
    outFile.close();
}

int main(){
    Mat img = imread("in.jpg");
    resize(img, img, Size(800, 800 * img.rows / img.cols));

    Mat out;
    Mat out2;
    add_noise(img, 30, out);

    const char *mode = "maf";
    const float fnum = 15.0;
    
    if (strcmp(mode, "maf") == 0){
        Mat h(fnum, fnum, CV_64F);
        h = 1.0 / (fnum * fnum);

        //フィルタをかける
        apply_filter(out, out2, h);
    }

    if (strcmp(mode, "gausian") == 0){
        Mat h = gausian(fnum);

        //フィルタをかける
        apply_filter(out, out2, h);
        //filter2D(out, out2, 8, h);
    }
    
    imshow("out", out2);
    //imwrite("out_maf.png", out2);
    std::ostringstream filename;
    filename << "out_" << fnum << mode <<".png";
    imwrite(filename.str(), out2);
    std::string filename1 = filename.str();
    psnr_text(out, out2, filename1);
    // waitKey(0);

}


