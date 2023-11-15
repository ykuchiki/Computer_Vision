#include <stdio.h>
#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace cv;

// 周波数領域の交換，低周波数が中央に来るように入れ替えてる
void exchange(Mat& img){
    int cx = img.cols / 2;
    int cy = img.rows / 2;

	Mat q0(img, Rect(0, 0, cx, cy));
	Mat q1(img, Rect(cx, 0, cx, cy));
	Mat q2(img, Rect(0, cy, cx, cy));
	Mat q3(img, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void DFT_img(Mat& img, Mat& complexI){
    Mat planes[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };
    // planesをcomplexIにマージする
    merge(planes, 2, complexI);
    // dft実行  complexIに結果が入る
    dft(complexI, complexI);
}

void util_psf(Mat& img, Mat& psf, Mat& out){
    Mat FI, FH;
    DFT_img(img, FI);
    DFT_img(psf, FH);

    for (int y = 0; y < FI.rows; y++) {
        for (int x = 0; x < FI.cols; x++) {
            Complexf & p1 = FI.at<Complexf>(y, x);
            Complexf & p2 = FH.at<Complexf>(y, x);
            p1 = p1 * p2;
        }
    }
    dft(FI, FI, DFT_INVERSE | DFT_SCALE);
    Mat planes[2];
    split(FI, planes);

    out = planes[0];
}


void iDFT_img(Mat& complexI, Mat& img) {
	Mat planes[2];
	// iDFT実行
	// dft実行  complexIに結果が入る
	dft(complexI, complexI, DFT_INVERSE | DFT_SCALE);
	// complexIの実部と虚部をplanesに分解
	split(complexI, planes);
	// 実部をコピー
	planes[0].copyTo(img);

}


// パワースペクトルの表示
void draw_powerspectral(Mat& img, Mat& out){
    if(img.channels() == 1) {
    Mat zeros = Mat::zeros(img.size(), CV_32F);
    Mat in[] = {img, zeros};
    merge(in, 2, img);
    }

    Mat planes[2];
    split(img, planes);  // 複素数の画像を実部と虚部に分割
    
	Mat magI;
	magnitude(planes[0], planes[1], magI);  // 実部と虚部から絶対値を計算
	// logをとる
	log(magI + 1, magI);
	double maxv;
	minMaxLoc(magI, NULL, &maxv);

    // magnitudeの画像を0から255に正規化する
    normalize(magI, out, 0, 255, NORM_MINMAX);
}


void util_filter(Mat& img, Mat& psf, Mat& out, const char* mode){
    Mat FI, FH;
	DFT_img(img, FI);
    DFT_img(psf, FH);
    exchange(FI);
    exchange(FH);
	
    
    if (strcmp(mode, "wiener") == 0){
        assert(FI.size() == FH.size() && "Sizes of FI and FH must be equal!");
        for (int y = 0; y < FI.rows; y++) {
		for (int x = 0; x < FI.cols; x++) {
			Complexf& p1 = FI.at<Complexf>(y, x);
			Complexf& p2 = FH.at<Complexf>(y, x);
			p1 = p1 * p2.conj() / (p2*p2.conj() + 0.0001f);
		}
	}
    }

    if (strcmp(mode, "inverse") == 0){
        for (int y = 0; y < FI.rows; y++) {
		for (int x = 0; x < FI.cols; x++) {
			Complexf& p1 = FI.at<Complexf>(y, x);
			Complexf& p2 = FH.at<Complexf>(y, x);
			p1 = p1 / (p2 + 0.0001f); // 定数の値を小さくするほどオリジナルに近づく
		}
	}
    }
    exchange(FI);
    iDFT_img(FI, out);

}

void image_restoration(Mat& img, Mat& psf, Mat& out, const char* mode){
    Mat tH = psf.clone();
	for (int y = 0; y < psf.rows; y++) {
		for (int x = 0; x < psf.cols; x++) {
			tH.at<float>(y, x) = psf.at<float>(psf.rows-1-y, psf.cols-1-x);
		}
	}

    // 推定すべき画像：初期値を0.5と一定の画像とする。
	Mat F(img.rows, img.cols, CV_32F), D;
	F = 0.5;

    for (int it = 0; it < 100; it++) {
        if (strcmp(mode, "grad") == 0){
            // 勾配を求める
            Mat Hf;
            filter2D(F, Hf, CV_32F, psf);   // 予測Fをpsfで畳み込む
            Mat D = (img - Hf), pD;         // 得られたHfとimgの差をとる
            double err = sum(D.mul(D))[0];
            //printf("it=%d, err=%.2f\n", it, err);
            printf("%.2f\n",err);
            filter2D(D, pD, CV_32F, tH);    // エラーDの勾配をpsfの反転であるtHを畳み込み計算
            // 更新
            double alpha = 0.1;
            F = F + alpha * pD;
            imshow("reconst", F);
		    waitKey(1);
        }
        if (strcmp(mode, "rl") == 0){
            // Richardson-Lucy
            Mat Hf;
            filter2D(F, Hf, CV_32F, psf);
            Mat D = img.mul(1.0f / Hf), pD;         // imgとHfの比率
            Mat diff = img - Hf;
            double err = sum(diff.mul(diff))[0];
            //printf("it=%d, MSE=%.2f\n", it, err);
            printf("%.2f\n",err);
            filter2D(D, pD, CV_32F, tH);
            F = F.mul(pD);
            imshow("reconst", F);
		    waitKey(1);
        }

    }

    out = F;

}

int main(){
    Mat img, psf, psf_;
    img = Mat_<float>(imread("in_genta.jpg", 0)) / 255.0;
    psf = Mat_<float>(imread("psf_genta.png", 0)) / 255.0;
    psf_ = psf.clone();

    //resize(img, img, Size(800, 800 * img.rows / img.cols));

    //センタリングした方が良さそう
	int padTop = (img.rows - psf.rows) / 2;
    int padBottom = img.rows - psf.rows - padTop;
    int padLeft = (img.cols - psf.cols) / 2;
    int padRight = img.cols - psf.cols - padLeft;
    copyMakeBorder(psf, psf, padTop, padBottom, padLeft, padRight, BORDER_CONSTANT, 0); 

    // フィルタの総和を1にする = 明るさ一定
    psf = psf / sum(psf);

    // Level 1
    // PSFで入力画像をフィルタにかける
    Mat G;
    //filter2D(img, G, -1, psf_);
    imwrite("./test.png", psf * 22255.0);
    util_psf(img, psf, G);
    exchange(G);
    imwrite("./util_psf.png", G * 255.0);


    // Level 2
    // DFTをかけ，低周波数成分が中心にくるように変換し，パワースペクトルを表示
    Mat complexI, complexII;
    DFT_img(G, complexI);
    DFT_img(psf, complexII);
    exchange(complexI);
    exchange(complexII);

    Mat out1, out2;
    draw_powerspectral(complexI, out1);
    draw_powerspectral(complexII, out2);
    imwrite("./power_spectral_img.png", out1);
    imwrite("./power_spectral_psf.png", out2);

    Mat out3;
    util_filter(G, psf, out3, "wiener");
    exchange(out3);
    imwrite("./util_wiener_img.png", out3 * 255.0);
    util_filter(G, psf, out3, "inverse");
    exchange(out3);
    imwrite("./util_inverse_img.png", out3 * 255.0);


    // Level 3
    // フーリエ変換を使わず，元の画像で推定を行う方法
    Mat out4;
    image_restoration(G, psf, out4, "grad");
    imwrite("./util_grad.png", out4 * 255.0);
    image_restoration(G, psf, out4, "rl");
    imwrite("./util_rl.png", out4 * 255.0);

}