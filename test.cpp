#include "liveness_api.h"
#include "face_detect_fb.h"
#include "opencv2/highgui.hpp"

using namespace face_detect_fb;

static void test_pics(const string imgfolder)
{
	std::vector<cv::String> filenames; // notice here that we are using the Opencv's embedded "String" class
	double test_hr = 0;

	cv::glob(imgfolder, filenames); // new function that does the job ;-)
	Ptr<SVM> model = liveness_load_SVM_model();

	int count = 0;
	for (unsigned int i = 0; i < filenames.size(); ++i)
	{
		Mat src = cv::imread(filenames[i].c_str(), IMREAD_COLOR);
		if (src.empty())
		{
			continue;
		}
		float r = liveness_test(src, model);
		if (r != 2.f)
		{
			//imshow("err", src);
			//waitKey(2000);
			//cout << r << endl;
		}

		r = std::abs(r - 2) <= FLT_EPSILON ? 1.f : 0.f;

		test_hr += r;
		count++;
	}
	test_hr = test_hr / count;
	return;
}

static void test_video()
{
	VideoCapture cap;
	//static const string fb_detect_model = "../public/detect_fb_tongyong.bin";
	static const string fb_detect_model = "detect_fb_tongyong.bin";
	const string mvPath = "F:/nazhi_Daniel/datas/SIW/movie/live/050-1-1-1-1.mov";
	fb_detect_instance<Mat> *ptr = nullptr;

	Ptr<SVM> model = liveness_load_SVM_model();
	if (model.empty())
	{
		return;
	}

	//cap.open(0);
	cap.open(mvPath);
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return;
	}

	face_detect_fb::fb_detect_param param1;

	FACE_ERROR_E err;
	err = face_detect_fb::fb_detect_init<Mat>(&ptr, &param1, fb_detect_model.c_str());
	if (FACE_OK != err)
	{
		return;
	}
	
	Mat frame;
	int image_count = 1;
	int max_faces_per_frame = 1;
	float predNum = 0;
	float correctNum = 0;
	float res = 0;
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break;

		vector<Mat> img;
		img.push_back(frame);
		vector<cv::Rect> faces;
		vector<float> scores;
		vector<int> faceNum;

		faces.resize(max_faces_per_frame * image_count);
		scores.resize(max_faces_per_frame * image_count);
		faceNum.assign(image_count, max_faces_per_frame);

		FACE_ERROR_E err2;
		err2 = face_detect_fb::fb_detect_face<Mat>(ptr, img.data(), image_count, faces.data(), scores.data(), faceNum.data());
		if (FACE_OK != err)
		{
			continue;
		}

		if (faces[0].x == 0 || faces[0].y == 0)
		{
			continue;
		}

		cv::rectangle(frame, faces[0], Scalar(0, 255, 0));

		Mat frame_roi = frame(faces[0]);

		imshow(" Demo", frame_roi);

		float r = liveness_test(frame_roi, model);
		cout << r << endl;
		predNum++;
		if (1 == r)
		{
			correctNum++;
		}
		res = 100 *(correctNum / predNum);
		cout << res << "%" << endl;

		char c = (char)waitKey(10);
		if (c == 27)
			break;
	}

	return;
}

int main()
{
	//cout << "OpenCV Version: " << CV_VERSION << endl;
	//test_pics("../data/validation/spoof");
	//return 0;

	test_video();
	return 0;

	//Ptr<SVM> model = liveness_load_SVM_model();
	//Mat img = imread("test.jpg", 1);
	//float r = 0;
	//r = liveness_test(img,  model);
	//if (1 == r)
	//	cout << "live" << endl;
	//else
	//	cout << "spoof" << endl;

	//return 0;
}
