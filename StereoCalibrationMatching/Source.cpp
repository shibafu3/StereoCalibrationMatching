#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Releaseモードの場合
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300.lib")

#pragma comment(lib, "C:\\Program Files\\Anaconda3\\libs\\python35.lib")
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

int main(){
	Mat Q;
	Mat mapx_l, mapy_l;
	Mat mapx_r, mapy_r;
	//xmlファイルからステレオキャリブレートに必要なマップデータを読み込み
	FileStorage cvfsr("C:\\Users\\0133752\\Desktop\\StereoCalibrate.xml", FileStorage::READ);
	FileNode node(cvfsr.fs, NULL);
	read(node["mapx_l"], mapx_l);
	read(node["mapy_l"], mapy_l);
	read(node["mapx_r"], mapx_r);
	read(node["mapy_r"], mapy_r);
	read(node["Q"], Q);


	//左右画像読み込み
	Mat image_l = imread("C:\\Users\\0133752\\Desktop\\l_00.jpg");
	Mat image_r = imread("C:\\Users\\0133752\\Desktop\\r_00.jpg");

	//ステレオカメラ用の補正マップを用いて入力画像を補正
	remap(image_l, image_l, mapx_l, mapy_l, INTER_LINEAR);
	remap(image_r, image_r, mapx_r, mapy_r, INTER_LINEAR);


	//ステレオマッチング用の各種パラメータ
	int window_size = 3;
	int minDisparity = 0;                          //minDisparity------------ - 最小視差値、一般には０で良い
	int numDisparities = 64;					   //numDisparities---------- - 最大視差値と最小視差値の差、１６の倍数にする
	int blockSize = 11;							   //SADWindowSize----------Sum of Absolute Differencesを計算するウィンドウのサイズ、３～１１の奇数を推奨
	int P1 = 8 * 3 * window_size * window_size;	   //P1 = 0 ---------------- - 値の決め方が示されているが、デフォルトのままで良かった
	int P2 = 32 * 3 * window_size * window_size;   //P2 = 0 ---------------- - 同上
	int disp12MaxDiff = 1;						   //disp12MaxDiff = 0 --------左右の視差の許容最大値、デフォルトはチェックしない、デフォルトで良かった
	int preFilterCap = 0;						   //preFilterCap = 0 ----------事前にフィルタで大きな視差をクリップする、デフォルトで良かった
	int uniquenessRatio = 10;					   //uniquenessRatio = 0 ------目的関数値の次点との差の％比率、０は比較しない意、デフォルトで十分であった
	int speckleWindowSize = 100;				   //speckleWindowSize = 0 ----小さい斑点やノイズを消すフィルタのサイズ、０のデフォルトは使用しない意、これは使って非常に効果があった。
	int speckleRange = 1;						   //speckleRange = 0 --------上記フィルタを使用するときの、視差の最大値、１～２が推奨で、１６倍される、１が良かった
	int fullDP = cv::StereoSGBM::MODE_SGBM;        //fullDP = false ------------フルスペックのダイナミックプログラミングを行わないのがデフォルト、trueにしても大差はなかった

	//ステレオBMのインスタンス化
	//StereoBMのcreateメソッドでポインタを取得
	//その際に各種パラメータを引数に入力
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(
		minDisparity,
		numDisparities,
		blockSize,
		P1,
		P2, disp12MaxDiff,
		preFilterCap,
		uniquenessRatio,
		speckleWindowSize, speckleRange, fullDP);

	//左右画像から深度マップを作成
	Mat disparity_sgbm;    //((cv::MatSize)leftImg.size, CV_16S);
	ssgbm->compute(image_l, image_r, disparity_sgbm);

	//カメラのQ行列を使ってdisparityマップを実距離の3次元マップに変換
	Mat _3dImage;
	reprojectImageTo3D(disparity_sgbm, _3dImage, Q);

	//深度マップを視覚的に分かるようにピクセル値を変換
	Mat disparity_map_sgbm;
	double min, max;
	//深度マップの最小値と最大値を求める
	minMaxLoc(disparity_sgbm, &min, &max);
	//CV_8UC1に変換、最大レンジを0～255にする
	disparity_sgbm.convertTo(disparity_map_sgbm, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));

	//結果表示
	imshow("image_l", image_l);
	imshow("image_r", image_r);
	imshow("result_sgbm", disparity_map_sgbm);
	waitKey(0);

	return 0;
}