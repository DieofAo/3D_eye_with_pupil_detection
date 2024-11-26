#include "PupilTrackingMethod.h"

using namespace std;
using namespace cv;

void PupilTrackingMethod::reset()
{
	previousPupils.clear();
	previousPupil = TrackedPupil();
	pupilDiameterKf.statePost.ptr<float>(0)[0] = 0.5*expectedFrameSize.width;
}

void PupilTrackingMethod::registerPupil(Pupil &pupil ) {
	Mat measurement = ( Mat_<float>(1,1) << pupil.majorAxis() );  //测量值只有一个float，只对瞳孔椭圆的长轴进行了kalman滤波，符合瞳孔放缩是准静态的假设。
	//if (predictedMaxPupilDiameter > 0) {
	//	float &majorAxis = measurement.ptr<float>(0)[0];
	//	if ( majorAxis > predictedMaxPupilDiameter) {
	//		pupil.clear();
	//		return;
	//	}
	//}

	if (pupil.confidence > minDetectionConfidence) {   //置信度较高，则更新瞳孔直径。
		previousPupil = TrackedPupil(pupil);
		previousPupils.emplace_back( previousPupil );  //
		pupilDiameterKf.correct(measurement);  //利用观测值，纠正预测值到最优值。measurement同时是返回值。
	} else
		previousPupil = TrackedPupil();

	//if (pupil.confidence > minDetectionConfidence) {
	//	previousPupil = TrackedPupil(ts, pupil);
	//	previousPupils.push_back( previousPupil );
	//} else
	//	previousPupil = TrackedPupil();
}

void PupilTrackingMethod::predictMaxPupilDiameter() {
	predictedMaxPupilDiameter = 1.5*pupilDiameterKf.predict().ptr<float>(0)[0]; //最大的瞳孔预测直径。
	if (previousPupils.size() < 20)  //双端队列中捕捉到的瞳孔数量少于20，则predictedMaxPupilDiameter复位置位最小。
		predictedMaxPupilDiameter = -1;
}

void PupilTrackingMethod::run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, PupilDetectionMethod &pupilDetectionMethod)
{
	cv::Size frameSize = { frame.cols, frame.rows };
	if (expectedFrameSize != frameSize ) {
		// Reference frame changed. Let's start over!
		expectedFrameSize = frameSize;
		reset();
	}

	// Remove old samples
	while (!previousPupils.empty()) {   //每次run，又瞳孔数据就清空，
			previousPupils.pop_front();
	}

	pupil.clear();
	predictMaxPupilDiameter();  //那么这个函数中的previousPupils.size()永远小于20，predictedMaxPupilDiameter永远等于-1. 这个和前面的每次run都清空历史数据有点冲突。

	if ( previousPupil.confidence == NO_CONFIDENCE ) {
		pupil = pupilDetectionMethod.runWithConfidence(frame, roi, -1, -1);  //调用类pupilDetectionMethod的静态函数
	} else {
		run(frame, roi, previousPupil, pupil);
	}

	registerPupil(pupil);//添加到瞳孔历史数据。
	return;
}
