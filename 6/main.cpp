#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;


struct obj_t {
	int     idx;
	Point2f pt;
};

void gen_template(vector<Mat> & Ts, char *name, vector<string> &labels) {

	for (int i = 0; i < strlen(name); i++) {
		Mat T = Mat(32, 32, CV_8U);
		T = 0;
		char buf[32];
		sprintf(buf, "%c", name[i]);
		putText(T, buf, Point(8, 24), FONT_HERSHEY_SIMPLEX, 0.7, 255, 2.0, LINE_AA);
		Ts.push_back(T);
        labels.push_back(string(1, name[i]));
	}
}

void gen_input_image(Mat & I, vector<Mat>& Ts, vector<obj_t>& objs) {

	int w, h;
	w = h = 512;
	I = Mat(h, w, CV_8U);
	I = 0;
	// paste random object & position
	for (int i = 0; i < 20; i++) {
		obj_t obj;
		obj.idx = rand()%Ts.size();
		obj.pt.x = (w- Ts[obj.idx].cols) * (rand() % 10000) / 10000.0;
		obj.pt.y = 32 + (h- Ts[obj.idx].rows - 32) * (rand() % 10000) / 10000.0;
		objs.push_back(obj);
		Mat A = (cv::Mat_<double>(2, 3) << 1, 0, obj.pt.x, 0, 1, obj.pt.y);
		Mat tmp;
		cv::warpAffine(Ts[obj.idx], tmp, A, I.size());
		I = I | tmp;
	}
}

// 2次データの二次曲面
void interpolate_quadrac_2D(
    float p1, float p2, float p3,
    float p4, float p5, float p6,
    float p7, float p8, float p9,
    float &sub_x, float &sub_y
){
    float b1,b2,b3,b4,b5,b6;
    float a,b,c,d,e;

    b1 = p1 + p3 + p4 + p6 + p7 + p9;
    b2 = p1 - p3 - p7 + p9;
    b3 = p1 + p2 + p3 + p7 + p8 +p9;
    b4 = -p1 + p3 - p4 + p6 - p7 + p9;
    b5 = -p1 - p2 - p3 + p7 + p8 + p9;
    b6 = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

    a = (3*b1-2*b6)/6.0;
    b = b2 / 4.0;
    c = (3*b3-2*b6)/6.0;
    d = b4/6.0;
    e = b5/6.0;

    sub_x = (2*c*d-b*e)/(b*b-4*a*c);
    sub_y = (2*a*e-b*d)/(b*b-4*a*c);
}

// 1次元のデータの2次曲線
float interpolate_1D(float p1, float p2, float p3){
    return 0.5*(p1 - p3) / (p1 - 2 * p2 + p3);
}

void detect(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res) {

	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);

		for (int y = 1; y < sim.rows -1; y++) {
			for (int x = 1; x < sim.cols-1; x++) {
				float p0 = sim.at<float>(y, x);
				float p1 = sim.at<float>(y, x-1);
				float p2 = sim.at<float>(y, x+1);
				float p3 = sim.at<float>(y-1, x);
				float p4 = sim.at<float>(y+1, x);
				// detect by threshold
				if (p0 > 0.7) {
					// add result
					obj_t obj;
					obj.pt.x = x;
					obj.pt.y = y;
					obj.idx = i;
					res.push_back(obj);
					printf("detect %d (%d, %d)\n", i, x, y);

					// Level 1
					// draw rectangle & ID
                    Point max_pt(x, y);   
                    int hw = Ts[i].cols;
                    int hh = Ts[i].rows;
                    rectangle(RGB, Rect(max_pt, max_pt + Point(hw, hh)), CV_RGB(0, 255, 0), 2, LINE_AA);
                    char ind_n[32];
                    snprintf(ind_n, sizeof(ind_n), "ID=%d", i);
                    putText(RGB, ind_n, Point(max_pt.x, max_pt.y - 10), FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255,255,0), 1.0, LINE_AA);
				}
			}
		}

	}
}

void NMS(const Mat& sim, vector<Point>& locations, float threshold){
    for (int y = 1; y < sim.rows - 1; y++) {
        for (int x = 1; x < sim.cols - 1; x++) {
            float value = sim.at<float>(y, x);
            if (value > threshold) {
                bool isMax = true;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        if (sim.at<float>(y + ky, x + kx) > value) {
                            isMax = false;
                            break;
                        }
                    }
                    if (!isMax) break;
                }
                if (isMax) {
                    locations.push_back(Point(x, y));
                }
            }
        }
    }

}

void detect_level2(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res, vector<obj_t>& objs) {

    int count_num=0;
    float count_sum=0;
	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;
        

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);
        vector<Point> locations;
        NMS(sim, locations, 0.8);
		for (const auto& loc : locations){
        // detect by threshold
            obj_t obj;
            obj.pt.x = loc.x;
            obj.pt.y = loc.y;
            obj.idx = i;
            res.push_back(obj);
            printf("detect_2 %d (%d, %d)\n", i, loc.x, loc.y);


            // objs(正解)との検索結果resの位置を比較
            for (const auto& correct_obj : objs) {
                float err_x = loc.x - correct_obj.pt.x;
                float err_y = loc.y - correct_obj.pt.y;
                float Euclid = sqrt(err_x * err_x + err_y * err_y);
                if (correct_obj.idx == i) {
                    if (Euclid < 3) {
                        count_sum = count_sum + Euclid;
                        printf("Euclid %.10f\n", count_sum);
                        count_num++;
                        //break; // Only count the closest match
                    }
                }
            }

            // Level 2
            // draw rectangle & ID
            Point max_pt(loc.x, loc.y);   
            int hw = Ts[i].cols;
            int hh = Ts[i].rows;
            rectangle(RGB, Rect(max_pt, max_pt + Point(hw, hh)), CV_RGB(0, 255, 0), 2, LINE_AA);
            char id_text[32];
            snprintf(id_text, sizeof(id_text), "ID=%d", i);
            putText(RGB, id_text, Point(max_pt.x, max_pt.y - 10), FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255,255,0), 1.0, LINE_AA);
		}
	}
    //検出率
    float stc_det =  static_cast<float>(count_num) / static_cast<float>(objs.size());
    char text[100];
    float dadan = static_cast<float>(count_sum);
    sprintf(text, "(prec = %.2f[%%]), mean_d = %.3f", stc_det * 100, dadan / count_num); // floatをstringに変換
    char num[100];
    sprintf(num, "%d/20 detected", count_num);
    putText(RGB, text, Point(150, 30), FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0,255,255), 1.0, LINE_AA);
    putText(RGB, num, Point(10, 30), FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0,255,255), 1.0, LINE_AA);
}

void detect_level3(Mat& I, Mat & RGB, vector<Mat>& Ts, vector<obj_t>& res, vector<obj_t>& objs) {

    int count_num=0;
    float count_sum=0;
	for (int i = 0; i < Ts.size(); i++) {

		Mat sim;
        

		matchTemplate(I, Ts[i], sim, cv::TM_CCOEFF_NORMED);
        vector<Point> locations;
        NMS(sim, locations, 0.8);
		for (const auto& loc : locations){
            float p1 = sim.at<float>(loc.y-1, loc.x-1);
            float p2 = sim.at<float>(loc.y-1, loc.x);
            float p3 = sim.at<float>(loc.y-1, loc.x+1);
            float p4 = sim.at<float>(loc.y, loc.x-1);
            float p5 = sim.at<float>(loc.y, loc.x);
            float p6 = sim.at<float>(loc.y, loc.x+1);
            float p7 = sim.at<float>(loc.y+1, loc.x-1);
            float p8 = sim.at<float>(loc.y+1, loc.x);
            float p9 = sim.at<float>(loc.y+1, loc.x+1);
            float sub_x, sub_y;
            interpolate_quadrac_2D(p1,p2,p3,p4,p5,p6,p7,p8,p9,sub_x,sub_y);
            //sub_x = interpolate_1D(p4,p5,p6);
            //sub_y = interpolate_1D(p2,p5,p8);

            // サブピクセルオフセットを元の座標に加算
            float pos_x = loc.x + sub_x;
            float pos_y = loc.y + sub_y;
        // detect by threshold
            obj_t obj;
            obj.pt.x = pos_x;
            obj.pt.y = pos_y;
            obj.idx = i;
            res.push_back(obj);
            printf("detect_2 %d (%f, %f)\n", i, pos_x, pos_y);


            // objs(正解)との検索結果resの位置を比較
            for (const auto& correct_obj : objs) {
                float err_x = pos_x - correct_obj.pt.x;
                float err_y = pos_y - correct_obj.pt.y;
                float Euclid = sqrt(err_x * err_x + err_y * err_y);
                if (correct_obj.idx == i) {
                    if (Euclid < 3) {
                        count_sum = count_sum + Euclid;
                        printf("Euclid %.10f\n", count_sum);
                        count_num++;
                        //break; // Only count the closest match
                    }
                }
            }

            // Level 3
            // draw rectangle & ID
            Point max_pt(loc.x, loc.y);   
            int hw = Ts[i].cols;
            int hh = Ts[i].rows;
            rectangle(RGB, Rect(max_pt, max_pt + Point(hw, hh)), CV_RGB(0, 255, 0), 2, LINE_AA);
            char id_text[32];
            snprintf(id_text, sizeof(id_text), "ID=%d", i);
            putText(RGB, id_text, Point(max_pt.x, max_pt.y - 10), FONT_HERSHEY_DUPLEX, 0.7, CV_RGB(255,255,0), 1.0, LINE_AA);
		}
	}
    //検出率
    float stc_det =  static_cast<float>(count_num) / static_cast<float>(objs.size());
    char text[100];
    float dadan = static_cast<float>(count_sum);
    sprintf(text, "(prec = %.2f[%%]), mean_d = %.3f", stc_det * 100, dadan / count_num); // floatをstringに変換
    char num[100];
    sprintf(num, "%d/20 detected", count_num);
    putText(RGB, text, Point(150, 30), FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0,255,255), 1.0, LINE_AA);
    putText(RGB, num, Point(10, 30), FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(0,255,255), 1.0, LINE_AA);
}



int main() {

	// gen tamplates: change your name with avoiding duplication. 
	vector<Mat> Ts;
    vector<string> labels;
	gen_template(Ts, "YutaroKchik", labels);

	// gen input image
	Mat I;
	vector<obj_t> objs;
	gen_input_image(I, Ts, objs);
	imshow("Input", I);
    imwrite("in.png", I);
    
	// run detection
	vector<obj_t> res;
	Mat RGB;
	cvtColor(I, RGB, COLOR_GRAY2BGR);
	detect(I, RGB, Ts, res);
    imwrite("level1.png", RGB);
    imshow("detect", RGB);

	// Level2 : Add Evaluation
    Mat RGB2;
    cvtColor(I, RGB2, COLOR_GRAY2BGR);
    detect_level2(I, RGB2, Ts, res, objs);
	//imshow("detect2", RGB2);
    imwrite("level2.png", RGB2);

    // Level3
    Mat RGB3;
    cvtColor(I, RGB3, COLOR_GRAY2BGR);
    detect_level3(I, RGB3, Ts, res, objs);
	imshow("detect2", RGB3);
    imwrite("level3.png", RGB3);

	while(1);
	waitKey(0);
}
