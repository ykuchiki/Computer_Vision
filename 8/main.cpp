#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

void test_siftmatch(Mat & img1, Mat& img2) {

	auto detector = SiftFeatureDetector::create(100);

	// detect SIFT keypoints
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	Mat rgb;
	drawKeypoints(img1, keypoints1, rgb, CV_RGB(255, 0, 0));
	//imshow("keypoint1", rgb);
    imwrite("level1-1.jpg", rgb);
	drawKeypoints(img2, keypoints2, rgb, CV_RGB(255, 0, 0));
	//imshow("keypoint2", rgb);
    imwrite("level1-2.jpg", rgb);

	// compute SIFT descriptor
	auto descriptor = SiftDescriptorExtractor::create();
	Mat descriptor1, descriptor2;
	descriptor->compute(img1, keypoints1, descriptor1);
	descriptor->compute(img2, keypoints2, descriptor2);

	// matching descriptor
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);

	// remove wrong matches
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++){
		if (matches[i].distance < 100.0){
			good_matches.push_back(matches[i]);
		}
	}
	// draw result
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, rgb);
    imwrite("level2.jpg", rgb);


}


void level3(Mat & img1, Mat& img2) {

	auto detector = SiftFeatureDetector::create(100);

	// detect SIFT keypoints
	vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	Mat rgb;
	drawKeypoints(img1, keypoints1, rgb, CV_RGB(255, 0, 0));
	drawKeypoints(img2, keypoints2, rgb, CV_RGB(255, 0, 0));

	// compute SIFT descriptor
	auto descriptor = SiftDescriptorExtractor::create();
	Mat descriptor1, descriptor2;
	descriptor->compute(img1, keypoints1, descriptor1);
	descriptor->compute(img2, keypoints2, descriptor2);


	// matching descriptor
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);

	// remove wrong matches
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++){
		if (matches[i].distance < 100.0){
			good_matches.push_back(matches[i]);
		}
	}
	// draw result
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, rgb);

    // 対応点をPoint2fのベクトルに変換
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
    points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
    points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

    // 射影行列（ホモグラフィ）を計算
    Mat H = cv::findHomography(points2, points1, RANSAC);

    // 画像を変形（warp）
    Mat warpedImg2;
    warpPerspective(img2, warpedImg2, H, cv::Size(img1.cols + img2.cols, img1.rows));
    imwrite("test.jpg", warpedImg2);

    // 結果保存用
    Mat result(cv::Size(img1.cols + img2.cols*2, img1.rows*2), img1.type());

    // imgを結果にコピー
    img1.copyTo(result(cv::Rect(0, 0, img1.cols, img1.rows)));

    // img1の右端の座標を準備
    vector<Point2f> srcPoints;
    srcPoints.push_back(Point2f(img1.cols, 0));

    // 変換された座標を格納するベクトル
    vector<Point2f> dstPoints;

    // 座標を変換
    perspectiveTransform(srcPoints, dstPoints, H);

    warpPerspective(img2, warpedImg2, H, cv::Size(img1.cols + img2.cols, img1.rows));

	// 重なり合う領域の幅を50ピクセルと仮定
    int overlapWidth = 50;

	// ブレンド処理
    for (int x = img1.cols - overlapWidth; x < img1.cols; x++) {
        int blendWidth = img1.cols - x;
        for (int y = 0; y < img1.rows; y++) {
            double alpha = static_cast<double>(blendWidth) / overlapWidth;
            Vec3b pixel1 = img1.at<Vec3b>(y, x);
            Vec3b pixel2 = warpedImg2.at<Vec3b>(y, x);

            // ピクセル値をブレンド
            Vec3b blendedPixel;
            for (int k = 0; k < 3; k++) {
                blendedPixel[k] = static_cast<uchar>(pixel1[k] * alpha + pixel2[k] * (1 - alpha));
            }
            warpedImg2.at<Vec3b>(y, x) = blendedPixel;
        }
    }

    // スティッチング結果を保存
    img1.copyTo(warpedImg2(cv::Rect(0, 0, img1.cols, img1.rows)));
    imwrite("level3.jpg", warpedImg2);
}




int main(){
    // 画像の読み込み
    Mat img1 = imread("input1.jpg");
    Mat img2 = imread("input2.jpg");

    // リサイズ
    resize(img1, img1, Size(800, img1.rows * 800 / img1.cols));
    resize(img2, img2, Size(800, img2.rows * 800 / img2.cols));

    //level1 & 2
	test_siftmatch(img1, img2);

    //level 3
    float s = 0.7, theta = 0.2;
	Mat A = (Mat_<double>(2, 3) << 
		 cos(theta) * s, sin(theta) * s,  30,
		-sin(theta) * s, cos(theta) * s, 140);
	warpAffine(img2, img2, A, img1.size());
    imwrite("level3-img2.jpg", img2);

    level3(img1,img2);

}