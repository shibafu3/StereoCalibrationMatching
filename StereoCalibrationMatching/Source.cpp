#ifdef _DEBUG
//Debug���[�h�̏ꍇ
#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_world300d.lib")            // opencv_core
#else
//Release���[�h�̏ꍇ
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
	//xml�t�@�C������X�e���I�L�����u���[�g�ɕK�v�ȃ}�b�v�f�[�^��ǂݍ���
	FileStorage cvfsr("C:\\Users\\0133752\\Desktop\\StereoCalibrate.xml", FileStorage::READ);
	FileNode node(cvfsr.fs, NULL);
	read(node["mapx_l"], mapx_l);
	read(node["mapy_l"], mapy_l);
	read(node["mapx_r"], mapx_r);
	read(node["mapy_r"], mapy_r);
	read(node["Q"], Q);


	//���E�摜�ǂݍ���
	Mat image_l = imread("C:\\Users\\0133752\\Desktop\\l_00.jpg");
	Mat image_r = imread("C:\\Users\\0133752\\Desktop\\r_00.jpg");

	//�X�e���I�J�����p�̕␳�}�b�v��p���ē��͉摜��␳
	remap(image_l, image_l, mapx_l, mapy_l, INTER_LINEAR);
	remap(image_r, image_r, mapx_r, mapy_r, INTER_LINEAR);


	//�X�e���I�}�b�`���O�p�̊e��p�����[�^
	int window_size = 3;
	int minDisparity = 0;                          //minDisparity------------ - �ŏ������l�A��ʂɂ͂O�ŗǂ�
	int numDisparities = 64;					   //numDisparities---------- - �ő压���l�ƍŏ������l�̍��A�P�U�̔{���ɂ���
	int blockSize = 11;							   //SADWindowSize----------Sum of Absolute Differences���v�Z����E�B���h�E�̃T�C�Y�A�R�`�P�P�̊�𐄏�
	int P1 = 8 * 3 * window_size * window_size;	   //P1 = 0 ---------------- - �l�̌��ߕ���������Ă��邪�A�f�t�H���g�̂܂܂ŗǂ�����
	int P2 = 32 * 3 * window_size * window_size;   //P2 = 0 ---------------- - ����
	int disp12MaxDiff = 1;						   //disp12MaxDiff = 0 --------���E�̎����̋��e�ő�l�A�f�t�H���g�̓`�F�b�N���Ȃ��A�f�t�H���g�ŗǂ�����
	int preFilterCap = 0;						   //preFilterCap = 0 ----------���O�Ƀt�B���^�ő傫�Ȏ������N���b�v����A�f�t�H���g�ŗǂ�����
	int uniquenessRatio = 10;					   //uniquenessRatio = 0 ------�ړI�֐��l�̎��_�Ƃ̍��́��䗦�A�O�͔�r���Ȃ��ӁA�f�t�H���g�ŏ\���ł�����
	int speckleWindowSize = 100;				   //speckleWindowSize = 0 ----���������_��m�C�Y�������t�B���^�̃T�C�Y�A�O�̃f�t�H���g�͎g�p���Ȃ��ӁA����͎g���Ĕ��Ɍ��ʂ��������B
	int speckleRange = 1;						   //speckleRange = 0 --------��L�t�B���^���g�p����Ƃ��́A�����̍ő�l�A�P�`�Q�������ŁA�P�U�{�����A�P���ǂ�����
	int fullDP = cv::StereoSGBM::MODE_SGBM;        //fullDP = false ------------�t���X�y�b�N�̃_�C�i�~�b�N�v���O���~���O���s��Ȃ��̂��f�t�H���g�Atrue�ɂ��Ă��卷�͂Ȃ�����

	//�X�e���IBM�̃C���X�^���X��
	//StereoBM��create���\�b�h�Ń|�C���^���擾
	//���̍ۂɊe��p�����[�^�������ɓ���
	cv::Ptr<cv::StereoSGBM> ssgbm = cv::StereoSGBM::create(
		minDisparity,
		numDisparities,
		blockSize,
		P1,
		P2, disp12MaxDiff,
		preFilterCap,
		uniquenessRatio,
		speckleWindowSize, speckleRange, fullDP);

	//���E�摜����[�x�}�b�v���쐬
	Mat disparity_sgbm;    //((cv::MatSize)leftImg.size, CV_16S);
	ssgbm->compute(image_l, image_r, disparity_sgbm);

	//�J������Q�s����g����disparity�}�b�v����������3�����}�b�v�ɕϊ�
	Mat _3dImage;
	reprojectImageTo3D(disparity_sgbm, _3dImage, Q);

	//�[�x�}�b�v�����o�I�ɕ�����悤�Ƀs�N�Z���l��ϊ�
	Mat disparity_map_sgbm;
	double min, max;
	//�[�x�}�b�v�̍ŏ��l�ƍő�l�����߂�
	minMaxLoc(disparity_sgbm, &min, &max);
	//CV_8UC1�ɕϊ��A�ő僌���W��0�`255�ɂ���
	disparity_sgbm.convertTo(disparity_map_sgbm, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));

	//���ʕ\��
	imshow("image_l", image_l);
	imshow("image_r", image_r);
	imshow("result_sgbm", disparity_map_sgbm);
	waitKey(0);

	return 0;
}