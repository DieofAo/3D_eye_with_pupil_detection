/*
 * Copyright (c) 2018, Thiago Santini
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for non-commercial purposes, without fee, and without a written
 * agreement is hereby granted, provided that:
 *
 * 1) the above copyright notice, this permission notice, and the subsequent
 * bibliographic references be included in all copies or substantial portions of
 * the software
 *
 * 2) the appropriate bibliographic references be made on related publications
 *
 * In this context, non-commercial means not intended for use towards commercial
 * advantage (e.g., as complement to or part of a product) or monetary
 * compensation. The copyright holder reserves the right to decide whether a
 * certain use classifies as commercial or not. For commercial use, please contact
 * the copyright holders.
 *
 * REFERENCES:
 *
 * Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci, PuRe: Robust pupil detection
 * for real-time pervasive eye tracking, Computer Vision and Image Understanding,
 * 2018, ISSN 1077-3142, https://doi.org/10.1016/j.cviu.2018.02.002.
 *
 *
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 * THE AUTHORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE AUTHORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE AUTHORS
 * HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS.
 */

#ifndef PURE_H
#define PURE_H

#include <bitset>
#include <random>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "PupilDetectionMethod.h"

using namespace std;

/*
类的成员变量：
std::vector<cv::Point> points;
cv::RotatedRect pointsMinAreaRect;
float minCurvatureRatio;
cv::RotatedRect outline;
cv::Rect pointsBoundingBox;
cv::Rect combinationRegion;
cv::Rect br;
cv::Rect boundaries;
cv::Point2f v[4];
cv::Rect outlineInscribedRect;
cv::Point2f mp;
float minorAxis, majorAxis;
float aspectRatio;
cv::Mat internalArea;
float innerMeanIntensity;
float outerMeanIntensity;
float contrast;
float outlineContrast;
float anchorDistribution;
float score;
std::bitset<4> anchorPointSlices;
cv::Scalar color;



类的方法：
bool isValid(const cv::Mat &intensityImage, const int &minPupilDiameterPx, const int &maxPupilDiameterPx, const int bias=5);
void estimateOutline();
bool isCurvatureValid();
float ratio(float a, float b) 
bool fastValidityCheck(const int &maxPupilDiameterPx);
bool validateAnchorDistribution();
bool validityCheck(const cv::Mat &intensityImage, const int &bias);
bool validateOutlineContrast(const cv::Mat &intensityImage, const int &bias);
bool drawOutlineContrast(const cv::Mat &intensityImage, const int &bias, string out);
void draw(cv::Mat out)
void draw(cv::Mat out, cv::Scalar color)
void drawit(cv::Mat out, cv::Scalar color)


*/


class PupilCandidate
{
public:
    std::vector<cv::Point> points;
    cv::RotatedRect pointsMinAreaRect;
    float minCurvatureRatio;

    cv::RotatedRect outline;

    cv::Rect pointsBoundingBox;
    cv::Rect combinationRegion;
    cv::Rect br;
    cv::Rect boundaries;
    cv::Point2f v[4];
    cv::Rect outlineInscribedRect;
    cv::Point2f mp;
    float minorAxis, majorAxis;
    float aspectRatio;
    cv::Mat internalArea;
    float innerMeanIntensity;
    float outerMeanIntensity;
    float contrast;
    float outlineContrast;
    float anchorDistribution;
    float score;
	std::bitset<4> anchorPointSlices;

	cv::Scalar color;

    enum {
        Q0 = 0,
        Q1 = 1,
        Q2 = 2,
        Q3 = 3,
    };

	PupilCandidate(std::vector<cv::Point> points) :
        minCurvatureRatio(0.198912f), // (1-cos(22.5))/sin(22.5)  ，一个45度圆弧的外截矩形的短边和长边的比值，也就是这个矩形的內截椭圆的长短轴的比，小于这个值的线段可以认为是近似直线，从而被忽略。

        anchorDistribution(0.0f),
        aspectRatio(0.0f),
        outlineContrast(0.0f),
		score(0.0f),
		color(0,255,0)
    {
        this->points = points;    //CV_CHAIN_APPROX_TC89_KCOS生成的用于近似原边缘的直线串组合。
    }
    bool isValid(const cv::Mat &intensityImage, const int &minPupilDiameterPx, const int &maxPupilDiameterPx, const int bias=5);
    void estimateOutline();
    bool isCurvatureValid();

    // Support functions
    float ratio(float a, float b) {
        std::pair<float,float> sorted = std::minmax(a,b);
        return sorted.first / sorted.second;
    }

    bool operator < (const PupilCandidate& c) const  //sort调用重载的符号函数<
    {
        return (score < c.score);
    }

    bool fastValidityCheck(const int &maxPupilDiameterPx);

    bool validateAnchorDistribution();

    bool validityCheck(const cv::Mat &intensityImage, const int &bias);

	bool validateOutlineContrast(const cv::Mat &intensityImage, const int &bias);
	bool drawOutlineContrast(const cv::Mat &intensityImage, const int &bias, string out);

    void updateScore()  //平均了各项指标的得分
    {
        score = 0.33*aspectRatio + 0.33*anchorDistribution + 0.34*outlineContrast;
        // ElSe style
        //score = (1-innerMeanIntensity)*(1+abs(outline.size.height-outline.size.width));
    }

    void draw(cv::Mat out){            //用直径为1的圆，当作点，将所有边界点画出来。
        //cv::ellipse(out, outline, cv::Scalar(0,255,0));
        //cv::rectangle(out, combinationRegion, cv::Scalar(0,255,255));
        for (unsigned int i=0; i<points.size(); i++)
            cv::circle(out, points[i], 1, cv::Scalar(0, 255, 255));

        cv::circle(out, mp, 3, cv::Scalar(0, 0, 255), -1);

		string s="unkonow string";
        //cv::putText(out, s.toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
        //cv::putText(out, QString::number(score,'g', 2).toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
        //cv::putText(out, QString::number(anchorDistribution,'g', 2).toStdString(), outline.center, CV_FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255));
    }

    void draw(cv::Mat out, cv::Scalar color){  //用直径为2的圆，将所有边界点画出来，并且用直线将他们连接起来
		int w = 2;
        cv::circle(out, points[0], w, color, -1);
        for (unsigned int i=1; i<points.size(); i++) {
            cv::circle(out, points[i], w, color, -1);
            cv::line(out, points[i-1], points[i], color, w-1);
        }
        cv::line(out, points[points.size()-1], points[0], color, w-1);
	}

	void drawit(cv::Mat out, cv::Scalar color){   //用直接为2的圆，将所有的边界点画出来，同时将矩形对应的椭圆画出来。
		int w = 2;
		for (unsigned int i=0; i<points.size(); i++)
			cv::circle(out, points[i], w, color, -1);
		cv::ellipse(out, outline, color);
	}

};

/*
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


子类的成员变量：
static std::string desc : 方法描述字符串
float meanCanthiDistanceMM;
float maxPupilDiameterMM;
float minPupilDiameterMM;
float meanIrisDiameterMM;
cv::RotatedRect detectedPupil;
cv::Size expectedFrameSize;
int outlineBias;
static const cv::RotatedRect invalidPupil;
cv::Size baseSize;
cv::Size workingSize;
float scalingRatio;
cv::Mat dx, dy, magnitude;
cv::Mat edgeType, edge;
cv::Mat input;
cv::Mat dbg;
int maxCanthiDistancePx;
int minCanthiDistancePx;
int maxPupilDiameterPx;
int minPupilDiameterPx;


子类的方法：
cv::RotatedRect run(const cv::Mat &frame) 
void run(const cv::Mat &frame, Pupil &pupil);
void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &userMinPupilDiameterPx=-1, const float &userMaxPupilDiameterPx=-1);
bool hasPupilOutline() 
bool hasConfidence()   父类有纯虚函数
bool hasCoarseLocation()   父类有纯虚函数
void init(const cv::Mat &frame);
void estimateParameters(int rows, int cols);
void detect(Pupil &pupil);
cv::Mat canny(const cv::Mat &in, bool blur=true, bool useL2=true, int bins=64, float nonEdgePixelsRatio=0.7f, float lowHighThresholdRatio=0.4f);
void filterEdges(cv::Mat &edges);
int pointHash(cv::Point p, int cols) { return p.y*cols+p.x;}
void removeDuplicates(std::vector<std::vector<cv::Point> > &curves, const int& cols) 
void findPupilEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
void combineEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
void searchInnerCandidates(std::vector<PupilCandidate> &candidates, PupilCandidate &candidate);
*/


//整个算法的核心，在于
/*
1、父类中通过harr角点，积分图，进行初步模板匹配找内黑外白的瞳孔，写了，但是没有在子类中用上。
2、自定义了canny，沿着边缘的法矢量方向，做了非极大抑制，对边缘分了2个级别，255强边缘，128的弱边缘。将强边缘以及和强边缘相邻的弱边缘，纳入考虑。孤立弱边缘直接丢弃。
3、对相邻的边缘点，形成线的过程，进行了自定义：短斜线直线化，边缘点密集处稀疏化，偏离直线的点就纠正回来，直角线断开化。
4、对以上得到的直线边缘，采用opencv中的findContour中CV_CHAIN_APPROX_TC89_KCOS，将边缘点用直线段经行矢量化。
5、定义了4检验瞳孔是否有效的标准：整幅图要包含椭圆的中心，边缘对比度条件，椭圆长短轴比，椭圆长短轴长度，前面3相的平均分。
6、对找到了的瞳孔的candidate 布尔分析。包括包含性。



*/
struct t_edgeFeatures
{
	int x;
	int y;
	float contrast;
	float lighterSideIntensity;
	float darkerSideIntensity;
};


class PuRe : public PupilDetectionMethod   //只是单次识别瞳孔，没有跟踪。
{
public:
    PuRe();
    ~PuRe();

    //自己加的一个保存当前瞳孔置信度
    float currentPupilConfidence;

    cv::RotatedRect run(const cv::Mat &frame) {                //外部调用运行时，调用这个run，返回瞳孔信息。
        Pupil pupil;
        currentPupilConfidence=0;
        run(frame, pupil);		
        return pupil;
    }

    void run(const cv::Mat &frame, Pupil &pupil);
	void run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &userMinPupilDiameterPx=-1, const float &userMaxPupilDiameterPx=-1);
	bool hasPupilOutline() { return true; }     //子类的函数，直接返回true，很诡异。
	bool hasConfidence() { return true; }  //父类中有纯虚函数。
	bool hasCoarseLocation() { return false; } //父类有纯虚函数。
	static std::string desc;

    float meanCanthiDistanceMM;
    float maxPupilDiameterMM;
    float minPupilDiameterMM;
    float meanIrisDiameterMM;
    cv::Point2i leftGilntPositionCurrentInRawImage; //还原到原始图中的glint坐标
    cv::Point2i rightGilntPositionCurrentInRawImage;

    std::vector<cv::Point2f> out23DEyeTracker;
protected:
    cv::RotatedRect detectedPupil;
    cv::Size expectedFrameSize;

    int outlineBias;

    static const cv::RotatedRect invalidPupil;

    /*
     *  Initialization
     */
    void init(const cv::Mat &frame);
    void estimateParameters(int rows, int cols);

    /*
     * Downscaling
     */
    cv::Size baseSize;
    cv::Size workingSize;
	float scalingRatio;

    /*
     *  Detection
     */
    void detect(Pupil &pupil, const cv::Mat &rawFrame);

    // Canny
	cv::Mat dx, dy, magnitude;   //dx：x方向偏导；dy：y方向偏导；magnitude：2D梯度模长。
    cv::Mat edgeType, edge;     //edgeType为边界类型，size和work size一样。  edge忽略了以弱边缘128构成的域，将强边缘255以及和强边缘相连的弱边缘加入了进来。
	cv::Mat canny(const cv::Mat &in, bool blur=true, bool useL2=true, int bins=64, float nonEdgePixelsRatio=0.7f, float lowHighThresholdRatio=0.4f);  

    // Edge filtering
	void filterEdges(cv::Mat &edges);  

	// Remove duplicates (e.g., from closed loops)
	int pointHash(cv::Point p, int cols) { return p.y*cols+p.x;}  //位置hash编码

	void removeDuplicates(std::vector<std::vector<cv::Point> > &curves, const int& cols) {   //删除有交集的，或者重复的轮廓。如果两个轮廓只是有交集，这个函数应该还是有问题。有种先来就占位，后来被删除。
		std::map<int,uchar> contourMap;  //保存的是curves中所有轮廓的起点。
		for (size_t i=curves.size(); i-->0;) {
			if (contourMap.count(pointHash(curves[i][0],cols)) > 0)   //第二步：如果某条轮廓的起点已经在map中，则从轮廓中删除这条轮廓。
				curves.erase(curves.begin()+i);
			else {
				for (int j=0; j<curves[i].size(); j++)           //第一步：将某一条contour上的所有点的唯一的hash值，都加入到一个contour map中。
					contourMap[pointHash(curves[i][j],cols)] = 1;
			}
		}
	}

    void findPupilEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
    void combineEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates);
	void searchInnerCandidates(std::vector<PupilCandidate> &candidates, PupilCandidate &candidate);

    cv::Mat input;
    cv::Mat dbg;

    int maxCanthiDistancePx;
    int minCanthiDistancePx;
    int maxPupilDiameterPx;
    int minPupilDiameterPx;


    //在PuRe基础上新加的内容，refine pupil 和 find glints
    cv::RotatedRect pupilOutline;
    cv::Point2i leftGilntPositionCurrent; //记录当前的瞬时glint位置
    cv::Point2i rightGilntPositionCurrent;
    double maxPupilSizeDuringTwoGlint;

    cv::Point2i leftGilntPositionReference; //记录最近一次两个glint都在的时候，glint的位置。
    cv::Point2i rightGilntPositionReference;
    bool referenceInitializedForBothGlintsFlag; //双glint初始化成功标志。
    long referenceInitalizedFrames;     //记录双glint已经初始化成功多少帧。要求每5min要初始化成功一次。 5*60*30，就是9000帧要重新初始化一次。

    void refine_pupil_and_find_glints(const cv::Mat& rawFrame,
                                      cv::Mat& input,
                                      PupilCandidate& selected,
                                      cv::Mat& detectedEdges,
                                      int contrastStep); //根据瞳孔提取的结果，进一步分析图像中的pupil和glints

    void recursive_greedy_connectivity_search(std::vector<cv::Point>& output,
                                              cv::Point& p,
                                              cv::Mat detectedEdges, 
                                              int contrastStep, 
                                              float darkerSideIntensityHighLimit,
                                              float lighterSideIntensityHighLimit, 
                                              float edgeContrastLowLimit);
    void direct_greedy_connectivity_search(std::vector<cv::Point>& output,
                                                 std::vector<cv::Point>& pointsSeeds,
                                                 const cv::Mat& detectedEdges, 
                                                 int contrastStep, 
                                                 float darkerSideIntensityHighLimit,
                                                 float lighterSideIntensityHighLimit,
                                                 float edgeContrastLowLimit);
};          

cv::Mat get_histogram_image(cv::Mat &image);  //得到直方图图像。
cv::MatND get_histogram(cv::Mat &image);  //得到直方图数组。

bool contrast_along_gradient(cv::Point& p,
                           cv::Mat& inputImage, 
						   cv::Mat& gradientX, 
						   cv::Mat& gradientY, 
						   int contrastStep,
						   float& meanContrast,
						   float& meanLighterSideIntensity,
						   float& meanDarkerSideIntensity);

void show_lines_random_color(cv::Mat& bg, 
                             bool drawLinesInBg,
                             std::vector<std::vector<cv::Point2i>>& lines, 
                             cv::String windowName);

void ratio_threshold_canny(cv::Mat& grayImageInput,
                           bool blurFlag,
                           bool useInputGradient,
						   int _blurSize,
						   float _blurSigam,
						   cv::Mat& blurred,
                           cv::Mat& inputGradient_x,
                           cv::Mat& inputGradient_y,
						   cv::Mat& inputMagnitude,
                           float notEdgePixRatio,
                           float lowHighThresholdRatio,
						   cv::Mat& outputEdge
                           );

void get_line_points_lists_from_edge(cv::Mat& edges, std::vector<std::vector<cv::Point>>& linePointsLists);
void point_mirror_about_axis(cv::Point& sourcePoint, cv::Point& mirroredPoint, cv::Point& axisPoint, float axisX, float axisY);

#endif // PURE_H
