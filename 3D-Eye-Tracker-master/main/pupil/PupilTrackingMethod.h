#ifndef PUPILTRACKINGMETHOD_H
#define PUPILTRACKINGMETHOD_H

#include <string>
#include <deque>

#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"

#include "PupilDetectionMethod.h"

//#include "utils.h"

/*
TrackedPupil和其父类Pupil相比，没有新加任何成员变量和成员函数。
*/
class TrackedPupil : public Pupil
{
public:
	TrackedPupil(const Pupil &pupil) :
		Pupil(pupil)      //子类的成员变量初始化列表和调用父类的构造函数，都是放在该方法名后面的:后。
	{}

	TrackedPupil() :
		Pupil()
	{}
};


/*
加了一个kalman滤波，考虑瞳孔直径的状态转移为1，即假设瞳孔直径静止为恒定值。模型转移矩阵方差很小，测量方差是模型转移矩阵方差的100倍。
成员变量：
mDesc，方法描述字符串；
previousPupil，上次瞳孔的状态；
previousPupils，之前瞳孔状态的线性deque表；
parallelDetection，并行检测标识符，false；
minDetectionConfidence，默认值0.7；
minTrackConfidence，默认值0.9；
pupilDiameterKf，瞳孔直径的kalman滤波cv::KalmanFilter类；
predictedMaxPupilDiameter，默认值-1；

成员函数：
run，重载两次；
description，返回方法描述字符串；
predictMaxPupilDiameter，预测最大瞳孔直径，如果历史数据少于20个，则取predictedMaxPupilDiameter为-1.
registerPupil，添加瞳孔数据到历史双端队列。
reset，重置
*/
class PupilTrackingMethod
{
public:
	PupilTrackingMethod() {
		pupilDiameterKf.init(1, 1);
		pupilDiameterKf.transitionMatrix = ( cv::Mat_<float>(1, 1) << 1 );
		cv::setIdentity( pupilDiameterKf.measurementMatrix );
		cv::setIdentity( pupilDiameterKf.processNoiseCov, cv::Scalar::all(1e-4) );
		cv::setIdentity( pupilDiameterKf.measurementNoiseCov, cv::Scalar::all(1e-2) );
		cv::setIdentity( pupilDiameterKf.errorCovPost, cv::Scalar::all(1e-1) );
	}
	~PupilTrackingMethod() {}

	// Tracking and detection logic
	void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, PupilDetectionMethod &pupilDetectionMethod);

	// Tracking implementation
	virtual void run(const cv::Mat &frame, const cv::Rect &roi, const Pupil &previousPupil, Pupil &pupil, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) = 0;

	std::string description() { return mDesc; }

private:

protected:
	std::string mDesc;

	cv::Size expectedFrameSize = {0, 0};
	TrackedPupil previousPupil;
	std::deque<TrackedPupil> previousPupils;

	bool parallelDetection = false;
	float minDetectionConfidence = 0.7f;
	float minTrackConfidence = 0.9f;

	cv::KalmanFilter pupilDiameterKf;
	float predictedMaxPupilDiameter = -1;

	void predictMaxPupilDiameter();
	void registerPupil(Pupil &pupil);

	void reset();
};

#endif // PUPILTRACKINGMETHOD_H
