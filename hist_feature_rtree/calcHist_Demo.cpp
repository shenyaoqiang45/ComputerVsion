/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define HIST_FEATURE_DIM  (256*3*2)
//const int hist_feat = 1536;
const int sample_num = 50000;

/**
 * @function main
 */
void test_hconcat()
{
	cv::Mat_<float> A = (cv::Mat_<float>(1, 2) << 1, 4);
	cv::Mat_<float> B = (cv::Mat_<float>(1, 2) << 7, 10);

	cv::Mat C;
	cv::hconcat(A, B, C);//即：将A加在B的左边。类似cv::hconcat(B, A, C)表示将A加在B的右边
	
	cout << C << endl;
	//C:
	//[1, 4, 7, 10;
	// 2, 5, 8, 11;
	// 3, 6, 9, 12]
	return;
}

int calcHistFeature(const cv::Mat &image, cv::Mat &feature)
{
	/// check image
	if (image.empty())
	{
		return -1;
	}

	if (image.cols == 0 || image.rows == 0)
	{
		return -1;
	}

	if (3 != image.channels())
	{
		return -1;
	}

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(image, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, 255, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, 255, NORM_MINMAX, -1, Mat());

	//cout << b_hist << endl;
	/// fea:768*1
	Mat fea;
	cv::vconcat(b_hist, g_hist, fea);
	cv::vconcat(fea, r_hist, feature);

	return 0;
}

int get_luv_yCrcb_hist_feature(const Mat &image, cv::Mat &feature)
{
	int ret = 0;
	/// check image
	if (image.empty())
	{
		return -1;
	}

	if (image.cols == 0 || image.rows == 0)
	{
		return -1;
	}

	if (3 != image.channels())
	{
		return -1;
	}
	//clock_t t0 = clock();
	Mat luv_image, yCrCb_image;
	cv::cvtColor(image, luv_image, CV_BGR2Luv);
	cv::cvtColor(image, yCrCb_image, CV_BGR2YCrCb);
	//clock_t t1 = clock();
	//std::cout << "get_luv_yCrcb_hist_feature<cpu_mat_t>执行时间：" << t1 - t0 << "ms" << endl;

	Mat luv_fea, yCrcb_fea;
	ret = calcHistFeature(luv_image, luv_fea);
	if (0 != ret)
	{
		return -1;
	}

	ret = calcHistFeature(yCrCb_image, yCrcb_fea);
	if (0 != ret)
	{
		return -1;
	}

	/// feature:1536*1
	cv::vconcat(luv_fea, yCrcb_fea, feature);
	return 0;
}

cv::Mat get_hist_featuret_for_rtree(const vector<Mat> &images)
{
	int ret = 0;
	Mat featureSet(images.size(), HIST_FEATURE_DIM, CV_32F);
	
	for (unsigned int i = 0; i < images.size(); i++)
	{
		//cv::imshow("h", data[i]);
		//cv::waitKey(0);
		//cout << i << endl;
		Mat feature;
		ret = get_luv_yCrcb_hist_feature(images[i], feature);
		if (0 != ret)
		{
			;
		}

		Mat row_fea = feature.reshape(0, 1);
		Mat row_i = featureSet.row(i);
		row_fea.convertTo(row_i, CV_32F);
	}

	return featureSet;
}

static std::vector<std::string> split(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;

	str += pattern;//扩展字符串以方便操作
	int size = str.size();

	for (int i = 0; i < size; i++)
	{
		pos = str.find(pattern, i);
		if (pos < size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

void read_imgList(const string& filename, vector<Mat>& images, vector<int>& labels) 
{
	std::ifstream file(filename.c_str(), ifstream::in);
	std::vector<std::string> result;
	string line;
	cv::Mat src;
	int iResult = 0;
	int count = 0;

	if (!file)
	{
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}

	while (getline(file, line))
	{
		result = split(line.c_str(), " ");

		string image_path = "E:\\backup_29_server\\liveness_data_collection\\128X128\\train_align_caffe_v1.0.7_ex2\\" + result[0];
		iResult = atoi(result[1].c_str());
		src = cv::imread(image_path, IMREAD_COLOR);
		if (src.empty())
		{
			cout << image_path << endl;
			continue;
		}

		images.push_back(src);

		labels.push_back(iResult);
		count++;
		if (0 == count % 2000)
		{
			cout << count << endl;
		}
		if (count == sample_num)
		{
			break;
		}
	}
	return;
}

static void test_and_save_classifier(const Ptr<StatModel>& model, const Mat& data, const Mat& responses,
										int ntrain_samples, int rdelta, const string& filename_to_save)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);

		float r = model->predict(sample);
		cout << r << endl;

		r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static bool build_rtrees_classifier(const string& data_filename,
									const string& filename_to_save,
									const string& filename_to_load)
{
	vector<Mat> images;	// vector to hold the images
	vector<int> labels;	// vector to hold the images
	int iSamples = 0;

	read_imgList(data_filename, images, labels);

	iSamples = images.size();
	Mat responses = Mat(iSamples, 1, CV_32S);
	memcpy(responses.data, labels.data(), labels.size()*sizeof(int));
	Mat dataRaw = get_hist_featuret_for_rtree(images);
	Mat data = dataRaw;

	Ptr<RTrees> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	// Create or load Random Trees classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<RTrees>(filename_to_load);
		if (model.empty())
			return false;
		ntrain_samples = 0;
	}
	else
	{
		// create classifier by using <data> and <responses>
		cout << "Training the classifier ...\n";
		//        Params( int maxDepth, int minSampleCount,
		//                   double regressionAccuracy, bool useSurrogates,
		//                   int maxCategories, const Mat& priors,
		//                   bool calcVarImportance, int nactiveVars,
		//                   TermCriteria termCrit );
		Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
		model = RTrees::create();
		model->setMaxDepth(10);
		model->setMinSampleCount(10);
		model->setRegressionAccuracy(0);
		model->setUseSurrogates(false);
		model->setMaxCategories(15);
		model->setPriors(Mat());
		model->setCalculateVarImportance(true);  
		model->setActiveVarCount(4);
		model->setTermCriteria(TC(100, 0.01f));
		model->train(tdata);
		cout << endl;
	}

	test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
	cout << "Number of trees: " << model->getRoots().size() << endl;

	// Print variable importance
	Mat var_importance = model->getVarImportance();
	if (!var_importance.empty())
	{
		double rt_imp_sum = sum(var_importance)[0];
		printf("var#\timportance (in %%):\n");
		int i, n = (int)var_importance.total();
		for (i = 0; i < n; i++)
			printf("%-2d\t%-4.1f\n", i, 100.f*var_importance.at<float>(i) / rt_imp_sum);
	}

	return true;
}

static bool build_svm_classifier(const string& data_filename,
								const string& filename_to_save,
								const string& filename_to_load)
{
	vector<Mat> images;	// vector to hold the images
	vector<int> labels;	// vector to hold the images
	int iSamples = 0;

	read_imgList(data_filename, images, labels);

	iSamples = images.size();
	Mat responses = Mat(iSamples, 1, CV_32S);
	memcpy(responses.data, labels.data(), labels.size()*sizeof(int));
	Mat dataRaw = get_hist_featuret_for_rtree(images);
	Mat data = dataRaw;

	Ptr<SVM> model;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	// Create or load Random Trees classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<SVM>(filename_to_load);
		if (model.empty())
			return false;
		ntrain_samples = 0;
	}
	else
	{
		// create classifier by using <data> and <responses>
		cout << "Training the classifier ...\n";
		Ptr<TrainData> tdata = prepare_train_data(data, responses, ntrain_samples);
		model = SVM::create();
		model->setType(SVM::C_SVC);
		model->setKernel(SVM::LINEAR);
		//model->setGamma(1);
		//model->setC(10);
		model->setC(1);
		model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
		model->train(tdata);
		cout << endl;
	}

	test_and_save_classifier(model, data, responses, ntrain_samples, 0, filename_to_save);
	return true;
}

void test_pics(const string imgfolder)
{
	std::vector<cv::String> filenames;
	int ret = 0;
	//string picPath = "E:\\data\\src_pics_20190117";

	cv::glob(imgfolder, filenames); // new function that does the job ;-)
	Ptr<RTrees> model;

	model = load_classifier<RTrees>("RTree_classifier_model.bin");
	if (model.empty())
		return;

	int count = 0;
	int total = filenames.size();
	for (unsigned int i = 0; i < total; ++i)
	{
		Mat src = imread(filenames[i], IMREAD_COLOR);
		if (src.empty())
		{
			continue;
		}
		//imshow("src", src);
		//waitKey(200);

		Mat feature;
		clock_t t0 = clock();
		ret = get_luv_yCrcb_hist_feature(src, feature);
		clock_t t1 = clock();
		std::cout << "monocular_live_detect_extract<cpu_mat_t>执行时间：" << t1 - t0 << "ms" << endl;
		if (0 != ret)
		{
			cout << "get_luv_yCrcb_hist_feature failed" << endl;;
		}
		Mat row_fea = feature.reshape(0, 1);


		float r = model->predict(row_fea);

		if (1 == r)
		{
			++count;
			continue;
		}

		imshow("src", src);
		waitKey(2000);
		cout << r << endl;

	}

	cout << "total num:" << total << endl;
	cout << "correct num:" << count << endl;
	return;
}

int main(int argc, char** argv)
{
	string filename_to_save = "RTree_classifier_modelnew.bin";
	string filename_to_load = "";
	string data_filename = "E://backup_29_server//liveness_data_collection//128X128//align_img_for_caffe_v1.0.7_ex2.txt";

	
	test_pics("E://data_collection//chongqing//0218//live");
	//test_pics("E://data_collection//live");
	//(void)build_rtrees_classifier(data_filename, filename_to_save, filename_to_load);

	//float d = 0,k = 0;

	////srand(time(0));
	//for (int i = 0; i < 10; i++)
	//{
	//	d = (rand() % 20 + 1) / (float)100.0;
	//	k = (rand() % 20 + 51) / (float)100.0;
	//	cout << d << " " << k << endl;
	//}

	return 0;

}
