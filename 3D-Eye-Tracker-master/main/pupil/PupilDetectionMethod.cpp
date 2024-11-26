#include "PupilDetectionMethod.h"

// TODO: clean up this interface and the one from the tracking
// this includes renaming everything from the old "run" method

using namespace std;
using namespace cv;

//#define DBG_COARSE_PUPIL_DETECTION
//#define DBG_OUTLINE_CONTRAST


/*
静态成员函数，PupilDetectionMethod类中没有静态成员变量，
所以这个函数Rect PupilDetectionMethod::coarsePupilDetection跟全局函数一样，计算用到的所有数据都依赖于形参且不能记住之前的计算结果。
coarsePupilDetection步骤：
1、先将图像缩放至工作分辨率， [frame.cols, frame.rows] → [workingWidth, workingHeight]，高度和宽度方向同比例缩放。 如原图frame的分辨率和working的分辨率不相似，则按照working分辨率的小边的比例缩放，大边的会比设定值小。
2、分别设定x和y方向的图像遍历步长，为 [图像对应边长1%, 1个像素] 中较大的那个
3、瞳孔半径假设，占对角线的比例在 [0.035, 0.145] 之间，半径搜索的步长为 [对角线的0.036，1个像素]中较大的那个， 半径搜索步长和最小半径相近。
4、3层循环扫描，利用harr 特征和积分图加速，扫描瞳孔半径和x，y轴的位置。
5、从已经检测到的瞳孔候选点中，筛选不超过瞳孔候选区域尺寸minCoverage的那些候选点。
6、返回利用harr 特征候选出来的瞳孔可能存在的roi区域。

*/
Rect PupilDetectionMethod::coarsePupilDetection(const Mat &frame, const float &minCoverage, const int &workingWidth, const int &workingHeight)
{
	// We can afford to work on a very small input for haar features, but retain the aspect ratio
	//步骤1，缩放图像
	float xr = frame.cols / (float) workingWidth;
	float yr = frame.rows / (float) workingHeight;
	float r = max( xr, yr );

	Mat downscaled;
    resize(frame, downscaled, Size(), 1/r, 1/r, CV_INTER_LINEAR);


	//步骤2，分别设定x和y方向的图像遍历步长，为 [图像对应边长1%, 1个像素] 中较大的那个
	int ystep = (int) max<float>( 0.01f*downscaled.rows, 1.0f);
	int xstep = (int) max<float>( 0.01f*downscaled.cols, 1.0f);

	float d = (float) sqrt( pow(downscaled.rows, 2) + pow(downscaled.cols, 2) );

	//步骤3，瞳孔半径假设，占对角线的比例在 [0.035, 0.145] 之间，半径搜索的步长为 [对角线的0.036，1个像素]中较大的那个， 半径搜索步长和最小半径相近。
	// Pupil radii is based on PuRe assumptions
	int min_r = (int) (0.5 * 0.07 * d);
	int max_r = (int) (0.5 * 0.29 * d);
	int r_step = (int) max<float>( 0.2f*(max_r + min_r), 1.0f);

	// TODO: padding so we consider the borders as well!

	/* Haar-like feature suggested by Swirski. For details, see
	 * Świrski, Lech, Andreas Bulling, and Neil Dodgson.
	 * "Robust real-time pupil tracking in highly off-axis images."
	 * Proceedings of the Symposium on Eye Tracking Research and Applications. ACM, 2012.
	 *
	 * However, we collect a per-pixel maxima instead of the global one
	*/
    //步骤4，在降采样的积分图上，通过harr特征，寻找和模板匹配最佳的点的位置，3层扫描，瞳孔半径和x，y轴的位置
	/*
	harr 特征
	a__________b
	 |   __   |
	 |  |__|  |
	d|________|c
	*/

	Mat itg;
	integral(downscaled, itg, CV_32S);
    Mat res = Mat::zeros( downscaled.rows, downscaled.cols, CV_32F);
	float best_response = std::numeric_limits<float>::min();
    deque< pair<Rect, float> > candidates;
	for (int r = min_r; r<=max_r; r+=r_step) {  //根据步骤3定义的半径和半径扫描步长，开始扫描不同大小的harr feature
		int step = 3*r;                         //harr feature的外环边长是内环边查的3倍，内环是黑色，外环是白色。

		Point ia, ib, ic, id;  //内环方形4个点
		Point oa, ob, oc, od;  //外环方形4个点

		int inner_count = (2*r) * (2*r);       //harr 特征内环面积
		int outer_count = (2*step)*(2*step) - inner_count; //harr特征外环面积

		float inner_norm = 1.0f / (255*inner_count);  //为求像素平均值，且规范化到[0, 1]做准备
		float outer_norm = 1.0f / (255*outer_count);

		for (int y = step; y<downscaled.rows-step; y+=ystep) { //harr 特征的y轴 位置的扫描，至少1个像素的移动步长扫描。
			oa.y = y - step;
			ob.y = y - step;
			oc.y = y + step;
			od.y = y + step;
			ia.y = y - r;
			ib.y = y - r;
			ic.y = y + r;
			id.y = y + r;
            for (int x = step; x<downscaled.cols-step; x+=xstep) { //harr 特征的x轴 位置的扫描，至少1个像素的移动步长扫描。
				oa.x = x - step;
				ob.x = x + step;
				oc.x = x + step;
				od.x = x - step;
				ia.x = x - r;
				ib.x = x + r;
				ic.x = x + r;
				id.x = x - r;
				int inner = itg.ptr<int>(ic.y)[ic.x] + itg.ptr<int>(ia.y)[ia.x] -itg.ptr<int>(ib.y)[ib.x] - itg.ptr<int>(id.y)[id.x]; //内方所有像素灰度值之和

				int outer = itg.ptr<int>(oc.y)[oc.x] + itg.ptr<int>(oa.y)[oa.x] -itg.ptr<int>(ob.y)[ob.x] - itg.ptr<int>(od.y)[od.x] - inner;//环形所有像素灰度值之和（外方形-内方形）

				float inner_mean = inner_norm*inner;
				float outer_mean = outer_norm*outer;
                float response = (outer_mean - inner_mean);  //外环平均值-内方平均值，作为harr匹配响应。

                if ( response < 0.5*best_response)
                    continue;

				if (response > best_response)       //迭代寻找最好的response的位置
					best_response = response;

				if ( response > res.ptr<float>(y)[x] ) {    //如果response比0大，或者比之前的记录的response大，则记录这个位置的瞳孔的尺度。
					res.ptr<float>(y)[x] = response;
					// The pupil is too small, the padding too large; we combine them.
					candidates.push_back( make_pair(Rect( 0.5*(ia+oa), 0.5*(ic+oc) ), response) ); //将某个位置的某个尺度下的response记录到candidate pair中。
				}

			}
		}
	}

	auto compare = [] (const pair<Rect, float> &a, const pair<Rect,float> &b) {  //Lambda 匿名函数表达式，用来调用 std里头的sort函数。
		return (a.second > b.second);
	};
	sort( candidates.begin(), candidates.end(), compare);

#ifdef DBG_COARSE_PUPIL_DETECTION
	Mat dbg;
	cvtColor(downscaled, dbg, CV_GRAY2BGR);
#endif

	// Now add until we reach the minimum coverage or run out of candidates

	//步骤5，从已经检测到的瞳孔候选点中，筛选不超过瞳孔候选区域尺寸minCoverage的那些候选点。
	Rect coarse;
	int minWidth = minCoverage * downscaled.cols;
	int minHeight = minCoverage * downscaled.rows;
	for ( int i=0; i<candidates.size(); i++ ) {
		auto &c = candidates[i];
		if (coarse.area() == 0)
			coarse = c.first;
		else
			coarse |= c.first;  //符号重载，矩形框求并集。
#ifdef DBG_COARSE_PUPIL_DETECTION
		rectangle(dbg, candidates[i].first, Scalar(0,255,255));
#endif
		if (coarse.width > minWidth && coarse.height > minHeight)   //如果已经探测到的候选区域，大于了设定的瞳孔在图像中的minCoverage，则停止进一步搜索。
			break;
    }

#ifdef DBG_COARSE_PUPIL_DETECTION
	rectangle(dbg, coarse, Scalar(0,255,0));
	resize(dbg, dbg, Size(), r, r);
	imshow("Coarse Detection Debug", dbg);
#endif


    // Upscale result  
	//步骤6，返回利用harr 特征候选出来的roi区域。
	coarse.x *= r;
	coarse.y *= r;
	coarse.width *= r;
	coarse.height *= r;

	// Sanity test
	Rect imRoi = Rect(0, 0, frame.cols, frame.rows);
	coarse &= imRoi;
	if (coarse.area() == 0)
		return imRoi;

	return coarse;
}

static const float sinTable[] = {
	0.0000000f  , 0.0174524f  , 0.0348995f  , 0.0523360f  , 0.0697565f  , 0.0871557f  ,
	0.1045285f  , 0.1218693f  , 0.1391731f  , 0.1564345f  , 0.1736482f  , 0.1908090f  ,
	0.2079117f  , 0.2249511f  , 0.2419219f  , 0.2588190f  , 0.2756374f  , 0.2923717f  ,
	0.3090170f  , 0.3255682f  , 0.3420201f  , 0.3583679f  , 0.3746066f  , 0.3907311f  ,
	0.4067366f  , 0.4226183f  , 0.4383711f  , 0.4539905f  , 0.4694716f  , 0.4848096f  ,
	0.5000000f  , 0.5150381f  , 0.5299193f  , 0.5446390f  , 0.5591929f  , 0.5735764f  ,
	0.5877853f  , 0.6018150f  , 0.6156615f  , 0.6293204f  , 0.6427876f  , 0.6560590f  ,
	0.6691306f  , 0.6819984f  , 0.6946584f  , 0.7071068f  , 0.7193398f  , 0.7313537f  ,
	0.7431448f  , 0.7547096f  , 0.7660444f  , 0.7771460f  , 0.7880108f  , 0.7986355f  ,
	0.8090170f  , 0.8191520f  , 0.8290376f  , 0.8386706f  , 0.8480481f  , 0.8571673f  ,
	0.8660254f  , 0.8746197f  , 0.8829476f  , 0.8910065f  , 0.8987940f  , 0.9063078f  ,
	0.9135455f  , 0.9205049f  , 0.9271839f  , 0.9335804f  , 0.9396926f  , 0.9455186f  ,
	0.9510565f  , 0.9563048f  , 0.9612617f  , 0.9659258f  , 0.9702957f  , 0.9743701f  ,
	0.9781476f  , 0.9816272f  , 0.9848078f  , 0.9876883f  , 0.9902681f  , 0.9925462f  ,
	0.9945219f  , 0.9961947f  , 0.9975641f  , 0.9986295f  , 0.9993908f  , 0.9998477f  ,
	1.0000000f  , 0.9998477f  , 0.9993908f  , 0.9986295f  , 0.9975641f  , 0.9961947f  ,
	0.9945219f  , 0.9925462f  , 0.9902681f  , 0.9876883f  , 0.9848078f  , 0.9816272f  ,
	0.9781476f  , 0.9743701f  , 0.9702957f  , 0.9659258f  , 0.9612617f  , 0.9563048f  ,
	0.9510565f  , 0.9455186f  , 0.9396926f  , 0.9335804f  , 0.9271839f  , 0.9205049f  ,
	0.9135455f  , 0.9063078f  , 0.8987940f  , 0.8910065f  , 0.8829476f  , 0.8746197f  ,
	0.8660254f  , 0.8571673f  , 0.8480481f  , 0.8386706f  , 0.8290376f  , 0.8191520f  ,
	0.8090170f  , 0.7986355f  , 0.7880108f  , 0.7771460f  , 0.7660444f  , 0.7547096f  ,
	0.7431448f  , 0.7313537f  , 0.7193398f  , 0.7071068f  , 0.6946584f  , 0.6819984f  ,
	0.6691306f  , 0.6560590f  , 0.6427876f  , 0.6293204f  , 0.6156615f  , 0.6018150f  ,
	0.5877853f  , 0.5735764f  , 0.5591929f  , 0.5446390f  , 0.5299193f  , 0.5150381f  ,
	0.5000000f  , 0.4848096f  , 0.4694716f  , 0.4539905f  , 0.4383711f  , 0.4226183f  ,
	0.4067366f  , 0.3907311f  , 0.3746066f  , 0.3583679f  , 0.3420201f  , 0.3255682f  ,
	0.3090170f  , 0.2923717f  , 0.2756374f  , 0.2588190f  , 0.2419219f  , 0.2249511f  ,
	0.2079117f  , 0.1908090f  , 0.1736482f  , 0.1564345f  , 0.1391731f  , 0.1218693f  ,
	0.1045285f  , 0.0871557f  , 0.0697565f  , 0.0523360f  , 0.0348995f  , 0.0174524f  ,
	0.0000000f  , -0.0174524f , -0.0348995f , -0.0523360f , -0.0697565f , -0.0871557f ,
	-0.1045285f , -0.1218693f , -0.1391731f , -0.1564345f , -0.1736482f , -0.1908090f ,
	-0.2079117f , -0.2249511f , -0.2419219f , -0.2588190f , -0.2756374f , -0.2923717f ,
	-0.3090170f , -0.3255682f , -0.3420201f , -0.3583679f , -0.3746066f , -0.3907311f ,
	-0.4067366f , -0.4226183f , -0.4383711f , -0.4539905f , -0.4694716f , -0.4848096f ,
	-0.5000000f , -0.5150381f , -0.5299193f , -0.5446390f , -0.5591929f , -0.5735764f ,
	-0.5877853f , -0.6018150f , -0.6156615f , -0.6293204f , -0.6427876f , -0.6560590f ,
	-0.6691306f , -0.6819984f , -0.6946584f , -0.7071068f , -0.7193398f , -0.7313537f ,
	-0.7431448f , -0.7547096f , -0.7660444f , -0.7771460f , -0.7880108f , -0.7986355f ,
	-0.8090170f , -0.8191520f , -0.8290376f , -0.8386706f , -0.8480481f , -0.8571673f ,
	-0.8660254f , -0.8746197f , -0.8829476f , -0.8910065f , -0.8987940f , -0.9063078f ,
	-0.9135455f , -0.9205049f , -0.9271839f , -0.9335804f , -0.9396926f , -0.9455186f ,
	-0.9510565f , -0.9563048f , -0.9612617f , -0.9659258f , -0.9702957f , -0.9743701f ,
	-0.9781476f , -0.9816272f , -0.9848078f , -0.9876883f , -0.9902681f , -0.9925462f ,
	-0.9945219f , -0.9961947f , -0.9975641f , -0.9986295f , -0.9993908f , -0.9998477f ,
	-1.0000000f , -0.9998477f , -0.9993908f , -0.9986295f , -0.9975641f , -0.9961947f ,
	-0.9945219f , -0.9925462f , -0.9902681f , -0.9876883f , -0.9848078f , -0.9816272f ,
	-0.9781476f , -0.9743701f , -0.9702957f , -0.9659258f , -0.9612617f , -0.9563048f ,
	-0.9510565f , -0.9455186f , -0.9396926f , -0.9335804f , -0.9271839f , -0.9205049f ,
	-0.9135455f , -0.9063078f , -0.8987940f , -0.8910065f , -0.8829476f , -0.8746197f ,
	-0.8660254f , -0.8571673f , -0.8480481f , -0.8386706f , -0.8290376f , -0.8191520f ,
	-0.8090170f , -0.7986355f , -0.7880108f , -0.7771460f , -0.7660444f , -0.7547096f ,
	-0.7431448f , -0.7313537f , -0.7193398f , -0.7071068f , -0.6946584f , -0.6819984f ,
	-0.6691306f , -0.6560590f , -0.6427876f , -0.6293204f , -0.6156615f , -0.6018150f ,
	-0.5877853f , -0.5735764f , -0.5591929f , -0.5446390f , -0.5299193f , -0.5150381f ,
	-0.5000000f , -0.4848096f , -0.4694716f , -0.4539905f , -0.4383711f , -0.4226183f ,
	-0.4067366f , -0.3907311f , -0.3746066f , -0.3583679f , -0.3420201f , -0.3255682f ,
	-0.3090170f , -0.2923717f , -0.2756374f , -0.2588190f , -0.2419219f , -0.2249511f ,
	-0.2079117f , -0.1908090f , -0.1736482f , -0.1564345f , -0.1391731f , -0.1218693f ,
	-0.1045285f , -0.0871557f , -0.0697565f , -0.0523360f , -0.0348995f , -0.0174524f ,
	-0.0000000f , 0.0174524f  , 0.0348995f  , 0.0523360f  , 0.0697565f  , 0.0871557f  ,
	0.1045285f  , 0.1218693f  , 0.1391731f  , 0.1564345f  , 0.1736482f  , 0.1908090f  ,
	0.2079117f  , 0.2249511f  , 0.2419219f  , 0.2588190f  , 0.2756374f  , 0.2923717f  ,
	0.3090170f  , 0.3255682f  , 0.3420201f  , 0.3583679f  , 0.3746066f  , 0.3907311f  ,
	0.4067366f  , 0.4226183f  , 0.4383711f  , 0.4539905f  , 0.4694716f  , 0.4848096f  ,
	0.5000000f  , 0.5150381f  , 0.5299193f  , 0.5446390f  , 0.5591929f  , 0.5735764f  ,
	0.5877853f  , 0.6018150f  , 0.6156615f  , 0.6293204f  , 0.6427876f  , 0.6560590f  ,
	0.6691306f  , 0.6819984f  , 0.6946584f  , 0.7071068f  , 0.7193398f  , 0.7313537f  ,
	0.7431448f  , 0.7547096f  , 0.7660444f  , 0.7771460f  , 0.7880108f  , 0.7986355f  ,
	0.8090170f  , 0.8191520f  , 0.8290376f  , 0.8386706f  , 0.8480481f  , 0.8571673f  ,
	0.8660254f  , 0.8746197f  , 0.8829476f  , 0.8910065f  , 0.8987940f  , 0.9063078f  ,
	0.9135455f  , 0.9205049f  , 0.9271839f  , 0.9335804f  , 0.9396926f  , 0.9455186f  ,
	0.9510565f  , 0.9563048f  , 0.9612617f  , 0.9659258f  , 0.9702957f  , 0.9743701f  ,
	0.9781476f  , 0.9816272f  , 0.9848078f  , 0.9876883f  , 0.9902681f  , 0.9925462f  ,
	0.9945219f  , 0.9961947f  , 0.9975641f  , 0.9986295f  , 0.9993908f  , 0.9998477f  ,
	1.0000000f
};

static void inline sincos(int angle, float& cosval, float& sinval)
{
	angle += (angle < 0 ? 360 : 0);
	sinval = sinTable[angle];
	cosval = sinTable[450 - angle];
}


//椭圆离散化插值。
static inline vector<Point> ellipse2Points(const RotatedRect &ellipse, const int &delta=1)
{
	int angle = ellipse.angle;

	// make sure angle is within range
	while( angle < 0 )
		angle += 360;
	while( angle > 360 )
		angle -= 360;

	float alpha, beta;
	sincos( angle, alpha, beta );

	double x, y;
	vector<Point> points;
	for( int i = 0; i < 360; i += delta )
	{
		x = 0.5*ellipse.size.width * sinTable[450-i];
		y = 0.5*ellipse.size.height * sinTable[i];
		points.push_back(
			Point( roundf(ellipse.center.x + x * alpha - y * beta),
				roundf(ellipse.center.y + x * beta + y * alpha) )
			);
	}
	return points;
}

/* Measures the confidence for a pupil based on the inner-outer contrast
 * from the pupil following PuRe. For details, see
 * Thiago Santini, Wolfgang Fuhl, Enkelejda Kasneci
 * "PuRe: Robust pupil detection for real-time pervasive eye tracking"
 * Under review on Elsevier's Computer Vision and Image Understanding journal.
 * TODO: update when published
 */


/*
静态函数，静态成员函数，PupilDetectionMethod类中没有静态成员变量
PupilDetectionMethod::outlineContrastConfidence跟全局函数一样，计算用到的所有数据都依赖于形参且不能记住之前的计算结果。

瞳孔边缘的对比度校验函数，将理论的瞳孔离散化，然后对比瞳孔内和瞳孔外一段距离内的像素的平局值之间是否有差异（求平局值不是除以像素个数，而是一个偏移距离），置信度1
将对比度达到一定成都的点数，除以椭圆圆周总点数，即为confidence

bias：瞳孔边缘外侧点的灰度值要比内侧点的灰度值大的差值
pupil：给定的瞳孔椭圆
frame：瞳孔灰度图

*/
float PupilDetectionMethod::outlineContrastConfidence(const Mat &frame, const Pupil &pupil, const int &bias)
{
	if ( ! pupil.hasOutline() )  //判断输入的pupil有效性
		return NO_CONFIDENCE;

	//获取瞳孔的基本描述信息
	//delta是
	Rect boundaries = { 0, 0, frame.cols, frame.rows };
	int minorAxis = min<int>(pupil.size.width, pupil.size.height);
	int delta = 0.15*minorAxis;   //用来对椭圆上的点向内外偏移的x轴或者y轴的距离。
	cv::Point c = pupil.center;

#ifdef DBG_OUTLINE_CONTRAST
	cv::Mat tmp;
	cv::cvtColor(frame, tmp, CV_GRAY2BGR);
	cv::ellipse(tmp, pupil, cv::Scalar(0,255,255));
	cv::Scalar lineColor;
#endif
	int evaluated = 0;
	int validCount = 0;

	//ellipse2Points(pupil, 10)，将瞳孔的椭圆，按照10度的扫描步长去离散为椭圆点序列。
	vector<Point> outlinePoints = ellipse2Points(pupil, 10);


	//遍历离散后的椭圆上的点，
	//dx，dy为椭圆边上的点到椭圆中心的距离
	//a为椭圆边上的点到椭圆中心的连线的斜率，这条线的方程为y=ax+b
	//b为上式种 y=kx+b 的y轴截距

	for (auto p=outlinePoints.begin(); p!=outlinePoints.end(); p++) {
		int dx = p->x - c.x;
		int dy = p->y - c.y;

		float a = 0;
		if (dx != 0)
			a = dy / (float) dx;
		float b = c.y - a*c.x;  //求每一个过椭圆中心和椭圆边界点的直线的斜率a和y轴截距b
		if (a == 0)   //斜率为0，则跳过该点。
			continue;

		if ( abs(dx) > abs(dy) ) {  //-45°~45°和135°~225°的范围内，单位dx带来的dy变化较小。
			int sx = p->x - delta;   //将椭圆边界上的点沿x轴方向内外偏移椭圆短轴长的0.15倍，下限
			int ex = p->x + delta;   //上限
			int sy = std::roundf(a*sx + b);  //根据椭圆边界上的该点和椭圆中心的连线方程，求出x内外偏移椭圆短轴长的0.15倍后的y轴坐标
			int ey = std::roundf(a*ex + b);
			cv::Point start = { sx, sy };      //建立偏移的两个点
			cv::Point end = { ex, ey };
			evaluated++;                     //待验证的点数加1

			if (!boundaries.contains(start) || !boundaries.contains(end) ) //图像范围内不包含偏移后的两个点，如果完整的检测到椭圆，这种情况几乎不可能。
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int x=sx; x<p->x; x++)
				m1 += frame.ptr<uchar>( (int) std::roundf(a*x+b) )[x]; //统计椭圆上的点到 偏移的一侧的点的之间的像素的灰度值的和
			m1 = std::roundf( m1 / delta );    //求平均

			m2 = count = 0;
			for (int x=p->x+1; x<=ex; x++) {
				m2 += frame.ptr<uchar>( (int) std::roundf(a*x+b) )[x];  //统计椭圆上的点到偏移的另一侧的点之间的像素灰度值之和
			}
			m2 = std::roundf( m2 / delta ); //求平局

#ifdef DBG_OUTLINE_CONTRAST
			lineColor = cv::Scalar(0,0,255);
#endif
			if (p->x < c.x) {// leftwise point
				if (m1 > m2+bias) {                      //椭圆上点在中心左侧是，m1就是椭圆外侧，m2是椭圆内侧
					validCount ++;                //椭圆外侧m1应该是白的，内侧m2是黑的，所以 m1要大于m2 一定的阈值，才算是有效瞳孔边缘，这个其实就是在校验瞳孔边缘的对比度。
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			} else {// rightwise point
				if (m2 > m1+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			}

#ifdef DBG_OUTLINE_CONTRAST
			cv::line(tmp, start, end, lineColor);
#endif
		}  
		else {           //45°~135°和225°~315°的范围内 ， 其实用向量的比例来控制偏移，就不需要考虑斜率k的问题，直接沿着椭圆径向内外偏移。
			int sy = p->y - delta;
			int ey = p->y + delta;
			int sx = std::roundf((sy - b)/a);
			int ex = std::roundf((ey - b)/a);
			cv::Point start = { sx, sy };
			cv::Point end = { ex, ey };

			evaluated++;
			if (!boundaries.contains(start) || !boundaries.contains(end) )
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int y=sy; y<p->y; y++)
				m1 += frame.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int y=p->y+1; y<=ey; y++)
				m2 += frame.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m2 = std::roundf( m2 / delta );

#ifdef DBG_OUTLINE_CONTRAST
			lineColor = cv::Scalar(0,0,255);
#endif
			if (p->y < c.y) {// upperwise point
				if (m1 > m2+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			} else {// bottomwise point
				if (m2 > m1+bias) {
					validCount++;
#ifdef DBG_OUTLINE_CONTRAST
					lineColor = cv::Scalar(0,255,0);
#endif
				}
			}

#ifdef DBG_OUTLINE_CONTRAST
			cv::line(tmp, start, end, lineColor);
#endif
		}
	}
	if (evaluated == 0)
		return 0;

#ifdef DBG_OUTLINE_CONTRAST
	cv::imshow("Outline Contrast Debug", tmp);
#endif

	return validCount / (float) evaluated;
}


/*
静态成员函数，只能访问静态成员变量，此类中没有静态成员变量，和普通全局函数一样
用来统计边缘点 points，环绕中心点 center，是否是4个象限均匀分布。 置信度2
这个函数有点不合理，对points，只记录了某个象限有没有点，而没有统计points中在每个象限中的点数    
*/

float PupilDetectionMethod::angularSpreadConfidence(const vector<Point> &points, const Point2f &center)
{
	enum {         //不同象限的标签
		Q0 = 0,
		Q1 = 1,
		Q2 = 2,
		Q3 = 3,
	};

	std::bitset<4> anchorPointSlices;   //申明一个包含4位的数据类型
	anchorPointSlices.reset();
	for (auto p=points.begin(); p!=points.end(); p++) {
		if (p->x - center.x < 0) {
			if (p->y - center.y < 0)
				anchorPointSlices.set(Q0);
			else
				anchorPointSlices.set(Q3);
		} else  {
			if (p->y - center.y < 0)
				anchorPointSlices.set(Q1);
			else
				anchorPointSlices.set(Q2);   //set函数是将某一位设置为1
		}
	}
	return anchorPointSlices.count() / (float) anchorPointSlices.size();  //count是求bitset中1的个数，size是bitset的大小，即总位数。
}


/*
静态成员函数，只能访问静态成员变量，此类中没有静态成员变量，和普通全局函数一样
椭圆的长短轴的比，置信度3
*/
float PupilDetectionMethod::aspectRatioConfidence(const Pupil &pupil)  //更具长短轴比来算圆置信度
{
	return pupil.minorAxis() / (float) pupil.majorAxis();
}


/*
椭圆边缘带内的图像边缘占椭圆周长的比，置信度4。
*/
float PupilDetectionMethod::edgeRatioConfidence(const Mat &edgeImage, const Pupil &pupil, vector<Point> &edgePoints, const int &band)
{
	if (!pupil.valid())
		return NO_CONFIDENCE;
	Mat outlineMask = Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8U);  //0值背景图
	ellipse(outlineMask, pupil, Scalar(255), band);  //band是画椭圆时的线的thickness，一个椭圆环，画在和图像一样大的0背景outlineMask上。
	Mat inBandEdges = edgeImage.clone();  //从提取的边缘，拷贝所有边缘
	inBandEdges.setTo(0, 255 - outlineMask); //255 - outlineMask形成一个除去band外的mask，将除了band内的边缘全部置零。
	findNonZero(inBandEdges, edgePoints);     //找到band内的非零边缘点
	return min<float>( edgePoints.size() / pupil.circumference(), 1.0 );  //求非零边缘的和椭圆周长之比。
}

