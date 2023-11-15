#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>

using namespace cv;


// [-1, +1]のガウスノイズの生成
float Box_Muller() {

	float X, Y;
	X = (rand() % 10000) / 10000.0;
	Y = (rand() % 10000) / 10000.0;

	return sqrt(-2 * log(X)) * cos(2 * 3.14159 * Y);
}


// 画像にノイズを加える
void add_noise_8U(Mat& img, Mat& out, float sigma) {
	img.copyTo(out);
	// ガウスノイズを加える
	for (int ptr = 0; ptr < img.cols * img.rows; ptr++) {
		int v = img.data[ptr] + Box_Muller() * sigma;
		if (v < 0)v = 0;
		if (v > 255)v = 255;
		out.data[ptr] = v;
	}
}


void gen_blob_ellipse_kadai(Mat& img, int seed){
	srand(seed);

	img = Mat(512, 512, CV_8U);

	img = 20;

	for (int i = 0; i < 30; i++) {
		float scale = 0.1 + (rand() % 80 + 20) / 100.0;
		ellipse(img, Point(rand() % 512, rand() % 512), Size(10 * scale, 20 * scale), rand() % 360, 0, 360, 140 + rand() % 100, -1, LINE_AA);
	}
	add_noise_8U(img, img, 20.0);

	//imshow("img", img);
    imwrite("./noise_plane.png", img);
}


void calc_histgram(Mat & img, float hist[]) {

	memset(hist, 0, sizeof(float) * 256);
	for (int ptr = 0; ptr < img.cols * img.rows; ptr++)
		hist[img.data[ptr]]++;

}

void closing(Mat& img,Mat& kernel){
    dilate(img, img, kernel);
    erode(img, img, kernel);
}

void opening(Mat& img, Mat& kernel){
    erode(img, img, kernel);
    dilate(img, img, kernel);
}


using namespace std;

void track(vector< vector<int> > & trees, int idx, int LUT[], int LN){
	if (LUT[idx] == 0) {
		LUT[idx] = LN;
		for (int k = 0; k < trees[idx].size(); k++) {
			track(trees, trees[idx][k], LUT, LN);
		}
	}
}



int compute_separation(float hist[]) {

	float max_sigma_b = 0;
	int max_T = 0;
	for (int T = 0; T <= 255; T++) {
		float m1 = 0, m2 = 0, w1 = 0, w2 = 0;
		int i;
		for (i = 0; i <= T; i++) {
			m1 += hist[i] * i;
			w1 += hist[i];
		}
		if(w1)m1 /= w1;
		for (; i <= 255; i++) {
			m2 += hist[i] * i;
			w2 += hist[i];
		}
		if(w2)m2 /= w2;

		float sigma_b = w1 * w2 * (m1-m2)* (m1 - m2);
		printf("T=%d, sigma_b=%.2f\n", T, sigma_b);
		if (max_sigma_b < sigma_b) {
			max_sigma_b = sigma_b;
			max_T = T;
		}
	}
	return max_T;
}


// 4近傍ラベリング
int labeling_4connect(Mat & img, Mat & Label) {
	uchar * in = img.data;
	int w = img.cols, h = img.rows;
	Label = Mat(h, w, CV_16U);
	Label = 0;
	ushort * label = (ushort*)Label.data;
	int LUT[10000] = { 0 };

	int label_index = 0;
	vector< vector<int> > trees(10000);

	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			int ptr = x + y * w;

			// 黒画素のみを処理
			if (in[ptr] == 255) {
				// 上の画素がラベルをもつとき
				int L1 = label[ptr - w];
				if (L1) {
					// 上の画素のラベルを注目画素に付ける。
					label[ptr] = L1;

					// 左の画素がラベルを持ち注目画素と異なるとき
					int L2 = label[ptr - 1];
					if (L2 && L1 != L2) {						
						trees[min(L2, L1)].push_back(max(L2, L1));
						trees[max(L2, L1)].push_back(min(L2, L1));
					}
				}
				else {
					// 注目画素の上の画素が白画素で左の画素がラベルをもつとき
					if (label[ptr - 1]) {
						// そのラベルを注目画素に付ける。
						label[ptr] = label[ptr - 1];
					}
					// 注目画素の上も左も白画素のとき
					else {
						label[ptr] = ++label_index;
					}
				}
			}
		}
	}

	int LN = 1;
	for (int i = 0; i < trees.size(); i++) {
		if (trees[i].size() > 0 && LUT[i] == 0) {
			track(trees, i, LUT, LN);
			LN++;
		}
	}
	// ルックアップテーブルを参照しながらラベルを再割り当て
	for (int ptr = 0; ptr < w * h; ptr++)
		label[ptr] = LUT[label[ptr]];

	return LN;
}


void coloring(Mat& img, int max_L, Mat& label, Mat &label_rgb){
    //Mat label_rgb = Mat(img.rows, img.cols, CV_8UC3);
	label_rgb = Mat(img.rows, img.cols, CV_8UC3);
	vector<Vec3b> lut(max_L);
	for (int i = 0; i < max_L; i++)
		lut[i] = Vec3b(rand() % 255, rand() % 255, rand() % 255);
	label_rgb = 0;
	for (int y = 0; y < label.rows; y++) {
		for (int x = 0; x < label.cols; x++) {
			int L = label.at<int>(y, x);
			if (L > 0) {
				label_rgb.at<Vec3b>(y, x) = lut[L];
			}
		}
	}

	//imshow("label", label_rgb);
    //img = label_rgb;
	//while(1)
	//waitKey(0);

}


vector<int> compute_area(Mat &label, int max_L){
    vector<int> areas(max_L, 0);

    // 面積計算
    for (int y = 0; y < label.rows; y++){
        for (int x = 0; x < label.cols; x++){
            int L = label.at<ushort>(y, x);  // (y, x)のラベルの値を取得，ushortはunsigned short型
            if (L > 0){
                areas[L]++;
            }
        }  
    }
    return areas;
}

//　バウンディングボックスの計算と描画，出力は個数
int compute_boundingbox(Mat &img, Mat &label, Mat &stats, int num){
    
    // Loop through all blobs
    for(int i = 1; i < stats.rows; i++){
        int area = stats.at<int>(i, CC_STAT_AREA);
        //printf("area:%d", area);
        if(area >= 300){
            int left = stats.at<int>(i, CC_STAT_LEFT);
            int top = stats.at<int>(i, CC_STAT_TOP);
            int width = stats.at<int>(i, CC_STAT_WIDTH);
            int height = stats.at<int>(i, CC_STAT_HEIGHT);
            
            rectangle(img, Point(left, top), Point(left + width, top + height), Scalar(255, 0, 0), 2);
            num++;
        }
    }
    return num;
}


void compute_moments(Mat &output, Mat &label, Mat &stats, Mat & centroids){
    
   for(int i = 1; i < stats.rows; i++){
    int area = stats.at<int>(i, CC_STAT_AREA);
    if(area >= 300){
        double x_c = centroids.at<double>(i, 0);
        double y_c = centroids.at<double>(i, 1);

        Mat blob = (label == i);  // 現在の連結成分だけを抽出
        Moments moment = moments(blob);
        double theta = 0.5 * std::atan2(2 * moment.mu11, moment.mu20 - moment.mu02);

        // 主軸方向のラインの描画
        int length = 50;  // ラインの長さ
        int end_x = int(x_c + length * std::cos(theta));
        int end_y = int(y_c + length * std::sin(theta));
        line(output, Point(int(x_c), int(y_c)), Point(end_x, end_y), Scalar(255, 255, 0), 2);
    }
    }

}

int main(){

    // level 1
    Mat plane;
    gen_blob_ellipse_kadai(plane, 8719);

    float hist[256];
    calc_histgram(plane, hist);
    int T = compute_separation(hist);

    // binalize
	Mat bin;
	threshold(plane, bin, T, 255, THRESH_BINARY);
	printf("T=%d\n", T);
	//imshow("bin", bin);
    imwrite("./bin.png", bin);


    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	closing(bin, kernel);
    opening(bin, kernel);

    imwrite("./test.png", bin);

    Mat label, stats, centroids;
	//int max_L = labeling_4connect(bin, label);
    int max_L = connectedComponentsWithStats(bin, label, stats, centroids, 4);
    
    Mat output;
    coloring(plane, max_L, label, output);


    imwrite("./level1.png", output);


    //Level 2
    int num = 0;
    num = compute_boundingbox(output, label, stats, num);
    //printf("%d", num);

    putText(output, "234-D8719", Point(15, 30), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255,255,0), 1.0, LINE_AA);
    String countstr = "N = " + to_string(num);
    int offsetX = 200;
    putText(output, countstr, Point(15 + offsetX, 30), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255,255,0), 1.0, LINE_AA);

    imwrite("./level2.png", output);


    //Level 3
    compute_moments(output, label, stats, centroids);

    imwrite("./level3.png", output);

}