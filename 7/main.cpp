#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;


void readImages(vector<Mat>& images) {
	for (int i = 0; i < 100; i++) {
		char file[1024];
		// sprintfはバッファオーバーフローのリスクあり
		// snprintfでバッファのサイズを指定することができる．
		snprintf(file, sizeof(file), "./faces/%06d.jpg", i + 1);
		// load as gray image
		Mat img = imread(file, 0);
		if (img.data)
		{
			img.convertTo(img, CV_32F, 1 / 255.0);
			images.push_back(img);
			// flip and add
			Mat imgFlip;
			flip(img, imgFlip, 1);
			//imwrite("da.png", imgFlip*255);
			images.push_back(imgFlip);
		}
	}
}
// Create data matrix from a vector of images
Mat createDataMatrix(const vector<Mat>& images)
{
	// data(N x pix)
	Mat data(images.size(), images[0].rows * images[0].cols, CV_32F);
	for (int i = 0; i < images.size(); i++)
	{
		Mat image = images[i].reshape(1, 1);
		image.copyTo(data.row(i));
	}
	return data;
}



void gen_face(vector<Mat> images, Mat & img, vector<int> & labels, int seed) {

	srand(seed);
	img = Mat(800, 1200, CV_32F);
	img = 1;
	int w = images[0].cols, h = images[0].rows;

	for (int y = 50; y <= 700; y += 250) {
		for (int x = 50; x <= 1000; x += 200) {
			int id = rand() % images.size();
			Mat roi(img, Rect(x, y, w, h));
			images[id].copyTo(roi);
			labels.push_back(id);
		}
	}
	img.convertTo(img, CV_8U, 255);
    imwrite("input.png", img);
	//imshow("img", img);
	//waitKey(0);
}

// sort rect according to position (left, top) : TV scan
bool fcomp(const Rect& a, const Rect& b) { return (a.y + a.x*0.1) < (b.y + b.x * 0.1); }

// level 1 & 3
void face_detect_crop(Mat & img, vector<Mat> & detect_faces) {
	CascadeClassifier cascade;
	cascade.load("haarcascade_frontalface_alt.xml");
	vector<Rect> faces;
	// run detect
	cascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(20, 20));
	sort(faces.begin(), faces.end(), fcomp);
	Mat rgb;
	cvtColor(img, rgb, COLOR_GRAY2BGR);
	for (int i = 0; i < faces.size(); i++)
	{
		int w = faces[i].width, h = faces[i].height;
		rectangle(rgb, Rect(faces[i].x, faces[i].y, w, h), Scalar(0, 0, 255), 3, LINE_AA);
		// crop head regeion
		//検出された顔の領域を拡大
        int padding = 40; // 顔の周りに追加するピクセル数
        int x = max(faces[i].x - padding, 0);
        int y = max(faces[i].y - padding, 0);
        int width = min(w + 2 * padding, img.cols - x);
        int height = min(h + 2 * padding, img.rows - y);
        Rect rect(x, y, width, height);
		// Rect rect = faces[i];
		Mat roi(img, rect), tmp;
		roi.copyTo(tmp);
		detect_faces.push_back(tmp);	}

	imwrite("level1.png", rgb);
	//imshow("detect face", rgb);
	//waitKey(0);
}


// 主成分分析を使って固有値を計算
// Level 2
void compute_eigenface(vector<Mat> & images, Mat& averageFace, vector<Mat>& eigenFaces) {
	//imshow("da", images[0]);
	//waitKey(0);
	Size sz = images[0].size();

	Mat data = createDataMatrix(images);
	int N = images.size();

	// compute PCA
	PCA pca(data, Mat(), PCA::DATA_AS_ROW, N);

	// average face
	averageFace = pca.mean.reshape(1, sz.height);

	// Find eigen vectors.
	Mat eigenVectors = pca.eigenvectors;

	// Reshape Eigenvectors to obtain EigenFaces
	for (int i = 0; i < N; i++)
	{
		Mat eigenFace = eigenVectors.row(i).reshape(1, sz.height);
		eigenFaces.push_back(eigenFace);
	}
	//imshow("ei",eigenFaces[0]);
	//waitKey(0);

}




// Level3
void level3(vector<Mat> & detect_faces, vector<Mat> & images, vector<int> labels, Mat &averageFace, vector<Mat> & eigenFaces) {
    int M = 40; // low-rank dimension
    int N = images.size();

    // gen dictionary of face database images
    Mat dict(N, M, CV_64F);
    for (int i = 0; i < N; i++) {
        for (int m = 0; m < M; m++) {
            dict.at<double>(i, m) = (images[i] - averageFace).dot(eigenFaces[m]);
        }
    }

    for (int i = 0; i < detect_faces.size(); i++) {
		if (detect_faces[i].empty()) {
            continue; // 空の画像はスキップ
        }

        Mat target;
        detect_faces[i].convertTo(target, CV_32F, 1/255.0);
        resize(target, target, averageFace.size());

        Mat reconst;

        // reconstruction
        averageFace.copyTo(reconst);
        Mat q(1, M, CV_64F);
        for (int m = 0; m < M; m++) {
            double a = (target - averageFace).dot(eigenFaces[m]);
            reconst += a * eigenFaces[m];
            q.at<double>(m) = a;
        }

        // 画像を表示する代わりにファイルに保存
        string target_filename = "target_" + to_string(i+1) + ".png";
        string reconst_filename = "reconst_" + to_string(i+1) + ".png";
        imwrite(target_filename, target * 255); // 元のスケールに戻す
        imwrite(reconst_filename, reconst * 255); // 元のスケールに戻す
    }
}


// level max
void test_recognition(vector<Mat> & detect_faces, vector<Mat> & images, vector<int> labels, Mat &averageFace, vector<Mat> & eigenFaces) {

	int M = 60;	// low-rank dimension
	int N = images.size();
	// gen dictionary of face database images
	Mat dict(N, M, CV_64F);

	// 固有顔と内積による係数の計算
	// 各切り出された顔画像detect_faces[i]に対してPCAによって得られた固有顔との内積を計算
	// この内積はその顔画像が固有顔のどの方向にどれだけ伸びてるかを示す
	// これらの係数は，顔画像の特徴を低次元で表現するために使用される
	for (int i = 0; i < N; i++) {
		for (int m = 0; m < M; m++) {
			dict.at<double>(i, m) = (images[i] - averageFace).dot(eigenFaces[m]);
		}
	}
        
	int correctCount = 0;
	for (int i = 0; i < detect_faces.size(); i++) {

		// 各画像に対して平均顔の形にリサイズ
		Mat target;
		detect_faces[i].convertTo(target, CV_32F, 1/255.0);
		resize(target, target, averageFace.size());

		Mat reconst;

		//imshow("target", target);
		// 再構成
		averageFace.copyTo(reconst);
		Mat q(1, M, CV_64F);
		for (int m = 0; m < M; m++) {
			// 平均顔から始めて各固有顔に対応する係数をかけたものを加算していく
			// これにより元の顔画像を固有顔の線形結合によって再構成する
			// 再構成された顔画像は元の顔画像の近似になる．
			double a = (target - averageFace).dot(eigenFaces[m]);
			reconst += a * eigenFaces[m];
			q.at<double>(m) = a;
		}

		//imshow("reconst", reconst);

		//*********************************************************
		// find nearest neighbor from database dict for query q
		double minDist = DBL_MAX;
        int minIndex = -1;
		// 再構成された顔画像の係数ベクトルqとデータベース内の各顔画像の係数ベクトルdictとの間でユークリッド距離を計算
		// 最も距離が小さい顔画像が切り出された顔画像に最も近いと判断される
        for (int j = 0; j < N; j++) {
            double dist = norm(q, dict.row(j));
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }

		// Check if the identified face matches the label
        if (labels[i] == minIndex) {
            correctCount++;
        }
    }

	double accuracy = static_cast<double>(correctCount) / detect_faces.size();
    cout << "Accuracy: " << accuracy << endl;
} 



int main() {

	Mat img,img2;

	// load database face
	vector<Mat> images;
	readImages(images);

	

	// gen test input image
	vector<int> labels;
	gen_face(images, img, labels, 8719);

	// level1
	vector<Mat> detect_faces;
	face_detect_crop(img, detect_faces);

	// level2
	Mat averageFace;
	vector<Mat> eigenFaces;
	compute_eigenface(images, averageFace, eigenFaces);

    imwrite("averageFace.png", averageFace*255);
    for (int i = 0; i < 4; ++i) {
		Mat eigenface;
		eigenFaces[i].copyTo(eigenface);
		normalize(eigenface, eigenface, 0, 255, NORM_MINMAX);
		eigenface.convertTo(eigenface, CV_8U);
        string filename = "eigenFace_" + to_string(i+1) + ".png";
        imwrite(filename, eigenface);
    }
    
	imwrite("test.png", detect_faces);
	// Level3
	vector<Mat> detect_faces3;
	level3(detect_faces, images, labels, averageFace, eigenFaces);

	// run recognition：level max
	test_recognition(detect_faces, images, labels, averageFace, eigenFaces);
}
