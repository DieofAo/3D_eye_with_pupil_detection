#ifndef PUPILDETECTIONMETHOD_H
#define PUPILDETECTIONMETHOD_H

#include <string>
#include <deque>
#include <bitset>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define NO_CONFIDENCE -1.0
#define SMALLER_THAN_NO_CONFIDENCE (NO_CONFIDENCE-1.0)
/*Pupil继承cv::RotatedRect，在cv::RotatedRect的基础上，添加了几个方法和一个成员变量confidence
1、对pupil数据重置clear（pupil的历史数据可以用来做跟踪）
2、x,y轴不同比例的resize
3、x,y轴同比例的resize
4、pupil平移shift，主要是对center的平移
5、校验检测到的pupil是否有效valid，主要是判断位置有效性和置信度要大于阈值
6、判断否是有轮廓的矩形hasOutline
7、返回矩形內截椭圆的长轴majorAxis，diameter
8、返回矩形內截椭圆的短轴minorAxis
9、返回矩形內截椭圆的周长circumference
*/
class Pupil : public cv::RotatedRect {            
public:
	Pupil(const RotatedRect &outline, const float &confidence) :
		RotatedRect(outline),
		confidence(confidence) {}

	Pupil(const RotatedRect &outline) :
		RotatedRect(outline),
		confidence(NO_CONFIDENCE) { }

	Pupil() { clear(); }

	float confidence;

	void clear() {
		angle = -1.0;
		center = { -1.0, -1.0 };
		size = { -1.0, -1.0 };
		confidence = NO_CONFIDENCE;
    }

	void resize(const float &xf, const float &yf) {
		center.x *= xf;
		center.y *= yf;
		size.width *= xf;
		size.height *= yf;
	}
	void resize(const float &f) {
		center *= f;
		size *= f;
	}
	void shift( cv::Point2f p ) { center += p; }

	bool valid(const double &confidenceThreshold=SMALLER_THAN_NO_CONFIDENCE) const {
		return center.x > 0 &&
			center.y > 0 &&
			size.width > 0 &&
			size.height > 0 &&
			confidence > confidenceThreshold;
	}

	bool hasOutline() const { return size.width > 0 && size.height > 0; }
	int majorAxis() const { return std::max<int>(size.width, size.height); }
	int minorAxis() const { return std::min<int>(size.width, size.height); }
	int diameter() const { return majorAxis(); }
	float circumference() const {
		float a = 0.5*majorAxis();
		float b = 0.5*minorAxis();
		return CV_PI * abs( 3*(a+b) - sqrt( 10*a*b + 3*( pow(a,2) + pow(b,2) ) ) );
	}
};


/*
构建一个用来 pupil detection的基类，然后根据具体的算法不同，派生出来不同的子类
派生的子类有：PuRe, ElSe, ExCuSe

父类留给子类要实现的接口（纯虚函数）：run，hasConfidence，hasCoarseLocation

父类成员变量：mDesc，方法描述字符串；

父类成员函数：
run，瞳孔识别，重载3种；  此函数的父类实现功能和 run(const cv::Mat &frame) 一样，要看子类的具体实现。
hasConfidence，纯虚函数；  
hasCoarseLocation，纯虚函数；
description，返回方法描述字符串；
runWithConfidence，调用run，并根据结果，给pupil中的confidence赋值；  此函数的父类实现功能和 run(const cv::Mat &frame) 一样，要看子类的具体实现。
getNextCandidate，将此次的结果，当作下次的候选，其实是一种kalman滤波，假设状态转移矩阵是1，待检测目标是静止的；

父类静态成员函数：（）
coarsePupilDetection，瞳孔检测的原始粗略结果； 这个函数功能相当于一个全局的瞳孔粗检测函数，非常重要。
outlineContrastConfidence，confidence的评估指标1，轮廓对比度；
edgeRatioConfidence，confidence的评估指标2，边缘比；
angularSpreadConfidence，confidence的评估指标3，角度延展；
aspectRatioConfidence，confidence的评估指标4，方向比；


静态成员变量：
静态成员变量在类的内部声明，声明时直接通过static关键字修饰
静态成员变量在类的外部定义与初始化，语法规则为Type ClassName::VarName = value，例如 int Test::c = 0 
静态成员变量不占用类实例化的对象和类大小，而是在类外（全局数据区）单独分配空间

静态成员函数：
静态成员函数属于整个类所有
可以通过类名和对象名访问public静态成员函数
静态成员函数只能访问静态成员变量和静态成员函数


                      |  静态成员函数   |  普通成员函数
所有类的对象共享        |     YES       |       YES
隐含this指针           |     NO        |       YES     
访问普通成员变量（函数） |     NO       |       YES
访问静态成员变量（函数） |     YES       |       YES
通过类名直接调用        |     YES       |       NO
通过对象名直接调用      |     YES       |       YES
*/
class PupilDetectionMethod
{
public:
    PupilDetectionMethod() {}
    ~PupilDetectionMethod() {}

    virtual cv::RotatedRect run(const cv::Mat &frame) = 0;
	virtual bool hasConfidence() = 0;
	virtual bool hasCoarseLocation() = 0;
	std::string description() { return mDesc; }

	//此函数的父类实现功能和 run(const cv::Mat &frame) 一样，要看子类的具体实现。
	virtual void run(const cv::Mat &frame, Pupil &pupil) {
        pupil.clear();
		pupil = run(frame);
		pupil.confidence = 1;
    }

	//此函数的父类实现功能和 run(const cv::Mat &frame) 一样，要看子类的具体实现。
	virtual void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) {
		/*
		(void) "variable name" 的功能是：当编译参数使用 -Wunused-parameter时，函数中有没有使用的变量，编译器就会对这些参数报warning。这个时候使用(void) "variable name"，就可以去掉编译的warning提示。
		当有多态时，父类的方法可能有些形参不需要使用，在子类的多态方法重新实现时，才会使用。那么在父类的方法可以使用(void) "variable name"，关掉形参未使用的编译warning提示。
		(void) 是一种没有任何操作的强制类型转换casting,其实相当于啥也没做。
		*/
		(void) roi;
		(void) minPupilDiameterPx;
		(void) maxPupilDiameterPx;
		run(frame, pupil);
	}

	////此函数的父类实现功能和 run(const cv::Mat &frame) 一样，要看子类的具体实现。
	// Pupil detection interface used in the tracking
	Pupil runWithConfidence(const cv::Mat &frame, const cv::Rect &roi, const float &minPupilDiameterPx=-1, const float &maxPupilDiameterPx=-1) {
		Pupil pupil;
		run(frame, roi, pupil, minPupilDiameterPx, maxPupilDiameterPx);
		if ( ! hasConfidence() )
			pupil.confidence = outlineContrastConfidence(frame, pupil);
		return pupil;
	}

	virtual Pupil getNextCandidate() { return Pupil(); }

	//以下都为静态函数，只能访问类的静态成员变量。
	// Generic coarse pupil detection
	static cv::Rect coarsePupilDetection(const cv::Mat &frame, const float &minCoverage=0.5f, const int &workingWidth=60, const int &workingHeight=40);    //这个函数功能相当于一个全局的瞳孔粗检测函数，非常重要。

	// Generic confidence metrics
	static float outlineContrastConfidence(const cv::Mat &frame, const Pupil &pupil, const int &bias=5);
	static float edgeRatioConfidence(const cv::Mat &edgeImage, const Pupil &pupil, std::vector<cv::Point> &edgePoints, const int &band=5);
	static float angularSpreadConfidence(const std::vector<cv::Point> &points, const cv::Point2f &center);
	static float aspectRatioConfidence(const Pupil &pupil);

	//Pupil test(const cv::Mat &frame, const cv::Rect &roi, Pupil pupil) { return pupil; }
protected:
	std::string mDesc;
};


#endif // PUPILDETECTIONMETHOD_H
