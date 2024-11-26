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

#include "PuRe.h"

#include <climits>
#include <iostream>
#include <opencv2/highgui.hpp>

//#define SAVE_ILLUSTRATION

using namespace std;
using namespace cv;

string PuRe::desc = "PuRe (Santini et. al 2018a)";

PuRe::PuRe() :
	//baseSize(320,240),  //默认将图像降到 320*240的分辨率进行瞳孔提取。
	baseSize(640,480),
	expectedFrameSize(-1,-1),
	outlineBias(5)
{
	mDesc = desc;

	/*
	 * 1) Canthi:
	 * Using measurements from white men
	 * Mean intercanthal distance 32.7 (2.4) mm   //两只眼睛的内眼角间距，即左眼的右角和右眼的左角之间的距离
	 * Mean palpebral fissure width 27.6 (1.9) mm  //眼裂，即同一只眼睛的两个眼角的距离。
	 * Jayanth Kunjur, T. Sabesan, V. Ilankovan
	 * Anthropometric analysis of eyebrows and eyelids:
	 * An inter-racial study
	 */
	meanCanthiDistanceMM = 27.6f;   //眼裂距离，同一只眼睛的两个眼角的距离。

	/*
	 * 2) Pupil:
	 * 2 to 4 mm in diameter in bright light to 4 to 8 mm in the dark
	 * Clinical Methods: The History, Physical, and Laboratory Examinations. 3rd edition.
	 * Chapter 58The Pupils
	 * Robert H. Spector.
	 */
	maxPupilDiameterMM = 8.0f;  //黑暗中4-8 mm
	minPupilDiameterMM = 2.0f;  //亮光中2-4 mm
	referenceInitializedForBothGlintsFlag=0;
	referenceInitalizedFrames=100000;
	maxPupilSizeDuringTwoGlint=-10000;
	
}

PuRe::~PuRe()
{
}

void PuRe::estimateParameters(int rows, int cols)
{
	/*
	 * Assumptions:
	 * 1) The image contains at least both eye corners
	 * 2) The image contains a maximum of 5cm of the face (i.e., ~= 2x canthi distance)
	 */
	float d = sqrt( pow(rows,2) + pow(cols,2) );
	maxCanthiDistancePx = d;             //假设一，图像至少包含眼睛的左右眼角，对角线长度为左右眼角距离的最大值。
	minCanthiDistancePx = 2*d/3.0;     //假设二，眼睛的两眼角距离至少对角线长度的2/3。
	//图像按照320*240，或者640*480，或者720*1080，眼睛水平的话，两眼角距离占图像水平宽度的55% ~ 83%   3/sqrt(13)

	maxPupilDiameterPx = maxCanthiDistancePx*(maxPupilDiameterMM/meanCanthiDistanceMM);  // 83% * 8/27.6
	minPupilDiameterPx = minCanthiDistancePx*(minPupilDiameterMM/meanCanthiDistanceMM);  // 55% * 2/27.6
}

void PuRe::init(const Mat &frame)
{
	if (expectedFrameSize == Size(frame.cols, frame.rows)) //第一次运行，会把expectedFrameSize设置为frame真实的size
		return;                                            //如果以后图像的大小变了，在重新设置。

	expectedFrameSize = Size(frame.cols, frame.rows);
	float rw = baseSize.width / (float) frame.cols;       //跟据expectedFrameSize 和 瞳孔识别时的baseSize ，来设置一个统一的横纵放缩比
	float rh = baseSize.height / (float) frame.rows;      //保证frame放缩后，横纵尺寸中有一个等于baseSize，另一个小于baseSize
	scalingRatio = min<float>( min<float>(rw, rh) , 1.0 );  //整体上保证相似性
}


/*
此处自定义的canny边缘提取，有4个步骤，得到的边缘供下一步使用。  相当于一个通用化的边缘提取工具。
1、高斯模糊滤波
2、通过梯度直方图，找到梯度值大于nonEdgePixelsRatio（0.7）时的高阈值，然后将高阈值的lowHighThresholdRatio（0.4）取为低阈值
3、将2D梯度方向分为45度一个域，总共8个域，沿着梯度方向，进行非极大抑制，得到两种边缘。大于高阈值的为255边缘，小于高阈值且大于低阈值的为128边缘
4、挑选得到边缘：将边缘为255的种子点和与它相连的点（边缘为128或者255），保存到了edge中。将孤立的或者成批的128的边缘忽略。
*/

Mat PuRe::canny(const Mat &in, bool blurImage, bool useL2, int bins, float nonEdgePixelsRatio, float lowHighThresholdRatio)  //自定义的canny求边缘方法， 忽略了128弱边缘形成了孤立连通域，保留了255强边缘以及与强边缘相连的弱边缘。
{
	(void) useL2;  //这个变量没有用
	/*
	 * Smoothing and directional derivatives
	 * TODO: adapt sizes to image size
	 */
	Mat blurred;
	if (blurImage) {  //默认blurImage为true
		Size blurSize(7,7);
		GaussianBlur(in, blurred, blurSize, 2, 2, BORDER_REPLICATE);   //高斯模糊
	} else
		blurred = in;

	Sobel(blurred, dx, dx.type(), 1, 0, 7, 1, BORDER_REPLICATE);   //求x方向偏导
	Sobel(blurred, dy, dy.type(), 0, 1, 7, 1, BORDER_REPLICATE);   //求y方向偏导

	/*
	 *  Magnitude
	 * 1、求梯度直方图
	 * 2、8方向梯度非极大抑制
	 */
	double minMag = 0;    //图像梯度最大值
	double maxMag = 0;    //图像梯度最小值
	float *p_res;                        //p_res：magnitude矩阵的某一行的首地址。
	float *p_x, *p_y; // result, x, y          p_x: x偏导矩阵某一行的向量首地址；p_y：y偏导矩阵某一行的向量首地址；

	cv::magnitude(dx, dy, magnitude);     //求图像2D梯度模长
	cv::minMaxLoc(magnitude, &minMag, &maxMag);  //找到图像梯度模长最大和最小的值

	/*
	 *  Threshold selection based on the magnitude histogram
	 */
	float low_th = 0;   //识别为边缘的下限梯度值，不是从直方图来，而是高阈值的40%
	float high_th = 0;  //梯度的强度是否识别为边缘的上限阈值

	// Normalization
	magnitude = magnitude / maxMag;  //归一化梯度。 https://www.zhihu.com/question/20467170

	// Histogram
	int *histogram = new int[bins]();     //梯度直方图
	Mat res_idx = (bins-1) * magnitude;  //将最大的梯度模长安排到数组索引最大的bin上。
	res_idx.convertTo(res_idx, CV_16U);
	short *p_res_idx=0;
	for(int i=0; i<res_idx.rows; i++){
		p_res_idx = res_idx.ptr<short>(i);
		for(int j=0; j<res_idx.cols; j++)
			histogram[ p_res_idx[j] ]++;
	}

	// Ratio
	int sum=0;
	int nonEdgePixels = nonEdgePixelsRatio * in.rows * in.cols;  //默认有70% 的梯度强度不够成为边缘，这是自适应边缘的一种方法。只有30%的像素位置的梯度能够成为边缘。
	for(int i=0; i<bins; i++){
		sum += histogram[i];
		if( sum > nonEdgePixels ){
			high_th = float(i+1) / bins ;
			break;
		}
	}
	low_th = lowHighThresholdRatio*high_th;  // 这个地方很诡异，low_th也是为边缘识别设立的么？降低了梯度成为边缘的要求。

	delete[] histogram;

	/*
	 *  Non maximum supression
	 * 将梯度在上半平面，分为45度角附近区域，135度角附近区域，0度角附近区域，90度角附近区域，4个区域，中间点的梯度要比沿着该区域直线的前后两点的梯度要大，就记录为边缘
	 * 大于高阈值记为255强边缘，大于低阈值的记为128若边缘。
	 */
	const float tg22_5 = 0.4142135623730950488016887242097f;
	const float tg67_5 = 2.4142135623730950488016887242097f;
	uchar *_edgeType;           //指向edgeType的矩阵的某一行的首地址。
	float *p_res_b, *p_res_t;   //梯度矩阵的某一行的上下两行的行向量的首地址。
	edgeType.setTo(0);
	for(int i=1; i<magnitude.rows-1; i++) {

		_edgeType = edgeType.ptr<uchar>(i);

		p_res=magnitude.ptr<float>(i);     //梯度矩阵某一行的首地址
		p_res_t=magnitude.ptr<float>(i-1);  //梯度矩阵前一行的首地址
		p_res_b=magnitude.ptr<float>(i+1);  //梯度矩阵后一行的首地址

		p_x=dx.ptr<float>(i);
		p_y=dy.ptr<float>(i);

		for(int j=1; j<magnitude.cols-1; j++){

			float m = p_res[j];
			if (m < low_th)          //当前点的梯度强度低于低限阈值，则不处理该点。
				continue;

			float iy = p_y[j];
			float ix = p_x[j];
			float y  = abs( (double) iy );
			float x  = abs( (double) ix );

			uchar val = p_res[j] > high_th ? 255 : 128;  //梯度强度大于高阈值，则为边界类型为255； 大于低阈值，小于高阈值，则边界类型为128。

			float tg22_5x = tg22_5 * x;
			if (y < tg22_5x) {                    //梯度方向在 0~22.5度范围内
				if (m > p_res[j-1] && m >= p_res[j+1])  //如果梯度值比左右的都大，记录该点的边界类型。
					_edgeType[j] = val;
			} else {
				float tg67_5x = tg67_5 * x;
				if (y > tg67_5x) {              //梯度方向在 67.5 ~90度
					if (m > p_res_b[j] && m >= p_res_t[j])  //如果梯度比上下的都大，则记录该点的边界类型。
						_edgeType[j] = val;
				} else {                        //梯度方向为 22.5~67.5度之间，
					if ( (iy<=0) == (ix<=0) ) {           //斜率为正
						if ( m > p_res_t[j-1] && m >= p_res_b[j+1])  //比上一行的前一个大，比下一行的后一个大，梯度的主方向为45度角附近。
							_edgeType[j] = val;
					} else {                             //斜率为负  
						if ( m > p_res_b[j-1] && m >= p_res_t[j+1])   //比下一行的前一个大，比上一行的后一个大，梯度的主方向为-45度角附近。
							_edgeType[j] = val;
					}
				}
			}
		}
	}

	/*
	 *  Hystheresis
	 * 
	 * 将边缘为255的种子点和与它相连的点（边缘为128或者255），保存到了edge中。
	 * 
	 * 将孤立的或者成批的128的边缘忽略。
	 */
	int pic_x=edgeType.cols;  //边缘类型宽
	int pic_y=edgeType.rows;  //边缘类型高
	int area = pic_x*pic_y;   //边缘图面积
	int lines_idx=0;
	int idx=0;  //不断加上pic_x，用来二维图像一维化访问过程中的换行。

	vector<int> lines;  //用来临时保存线段。
	edge.setTo(0);  //最终保存数据结果。
	for(int i=1;i<pic_y-1;i++){               //外围留了一圈，一个像素。
		for(int j=1;j<pic_x-1;j++){           //外围留了一圈，一个像素。

			if( edgeType.data[idx+j] != 255 || edge.data[idx+j] != 0 )   // edgeType.data[idx+j] == 255 &&  edge.data[idx+j] == 0, 就不会continue，满足条件，执行循环体。 忽略了弱边缘128的种子点。
				continue;

			edge.data[idx+j] = 255;   //edgeType.data[idx+j] == 255 &&  edge.data[idx+j] == 0才会执行这一步。 相当于是强边缘，直接记录为边。idx中已经包含了行的整数倍。  
			lines_idx = 1;
			lines.clear();
			lines.push_back(idx+j);  //当前线段上记录这一个点
			int akt_idx = 0;

			while(akt_idx<lines_idx){   //相当于是在做一个米字形生长，以一个种子点开始，只要旁边8个点满足条件，就加入到直线中来。
				int akt_pos=lines[akt_idx];
				akt_idx++;

				if( akt_pos-pic_x-1 < 0 || akt_pos+pic_x+1 >= area )  //akt_pos超出图像处于，就提前结束循环。左后方的点线性坐标小于0,或者右前方的点，线性坐标大于总像素。
					continue;

				for(int k1=-1;k1<2;k1++)         //对某一个点进行米字型连接分析，如果边界数据不  edge.data[(akt_pos+(k1*pic_x))+k2] ==0 && edgeType.data[(akt_pos+(k1*pic_x))+k2] != 0 就进行后续分析
					for(int k2=-1;k2<2;k2++){
						if(edge.data[(akt_pos+(k1*pic_x))+k2]!=0 || edgeType.data[(akt_pos+(k1*pic_x))+k2]==0)
							continue;
						edge.data[(akt_pos+(k1*pic_x))+k2] = 255;   //将此点的边界置为255，
						lines.push_back((akt_pos+(k1*pic_x))+k2);   //将此点加入到线的vector中。
						lines_idx++;
					}
			}
		}
		idx+=pic_x;  //换到下一行
	}

	return edge;
}

/*
这里输入的edge就是上个函数中输出的edge： 忽略了弱边缘形成了孤立连通域，保留了强边缘以及与强边缘相连的弱边缘。
                                      在梯度方向上，有非极大抑制的基础上。也就是梯度线条的切向不受影响，法向会被非极大抑制，相当于对模糊线条进行了瘦身，精确定位。
opencv的图像坐标系为，原点位于左上角，x轴向右轴，y垂直向下
opencv中mat的data的2D矩阵线性访问时，内存的排列顺序是：先排通道，再排行
这个函数的功能是：
1、短斜线直线化
2、边缘点临近过多，就稀疏化，如果某一个点的邻居有3个点以上，就将该点从边缘中踢除。
3、如果有的直线边，有一个点偏离直线，形成折线，就把他纠正回来。
4、将长的直角线段断开。
*/
void PuRe::filterEdges(cv::Mat &edges)    //相当于对边缘去噪滤波，通用工具。
{
	// TODO: there is room for improvement here; however, it is prone to small
	// mistakes; will be done when we have time
	int start_x = 5;
	int start_y = 5;
	int end_x = edges.cols - 5;
	int end_y = edges.rows - 5;

	//斜直线进行直线化。
	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar box[9];

			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){

				/*
				米字型9个像素领域分析，将4从边缘中踢除
				box[9]
				     1
                 3   4   5
					 7
				*/

				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];

                //斜折线进行直线化
				if((box[5] && box[7])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[5] && box[1])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[3] && box[7])) edges.data[(edges.cols*(j))+(i)]=0;
				if((box[3] && box[1])) edges.data[(edges.cols*(j))+(i)]=0;

			}
		}

	//too many neigbours
	//如果一个点的8领域内，属于边缘的像素点数目超过3个，就将该点从边缘中剔除。
	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar neig=0;

			for(int k1=-1;k1<2;k1++)
				for(int k2=-1;k2<2;k2++){

					if(edges.data[(edges.cols*(j+k1))+(i+k2)]>0)
						neig++;
				}

			if(neig>3)
				edges.data[(edges.cols*(j))+(i)]=0;

		}

	/*
	水平直线或者垂直中，有一个点或者2个点偏离该直线，则将他们纠正回来。
	*/
	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){
			uchar box[17];

			/*
			点4为中心点
			0    1     2    12
			3    4     5    9     11
			6    7     8    13
			15   10    16 
			     14
			*/
			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){
				box[0]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[2]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];

				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];

				box[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];
				box[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];

				//external
				box[9]=(uchar)edges.data[(edges.cols*(j))+(i+2)];
				box[10]=(uchar)edges.data[(edges.cols*(j+2))+(i)];


				box[11]=(uchar)edges.data[(edges.cols*(j))+(i+3)];
				box[12]=(uchar)edges.data[(edges.cols*(j-1))+(i+2)];
				box[13]=(uchar)edges.data[(edges.cols*(j+1))+(i+2)];


				box[14]=(uchar)edges.data[(edges.cols*(j+3))+(i)];
				box[15]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box[16]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];

				//垂直的折线，直线化。4 和 10 之间，有一个点，6或者8跳出直线
				if( (box[10] && !box[7]) && (box[8] || box[6]) ){
					edges.data[(edges.cols*(j+1))+(i-1)]=0;
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i)]=255;
				}

				//垂直的折线，直线化。4和14之间，有两个点跳出，6或者8中间的一个，15或者16中的一个。
				if( (box[14] && !box[7] && !box[10]) && ( (box[8] || box[6]) && (box[16] || box[15]) ) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i-1)]=0;
					edges.data[(edges.cols*(j+2))+(i+1)]=0;
					edges.data[(edges.cols*(j+2))+(i-1)]=0;
					edges.data[(edges.cols*(j+1))+(i)]=255;
					edges.data[(edges.cols*(j+2))+(i)]=255;
				}

				//水平折线直线化。4和9之间，有一个点跳出，2或者8
				if( (box[9] && !box[5]) && (box[8] || box[2]) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j-1))+(i+1)]=0;
					edges.data[(edges.cols*(j))+(i+1)]=255;
				}

				//水平折线直线化，4和11之间，有两个点跳出，2或者8中间的一个，12或者13中的一个。
				if( (box[11] && !box[5] && !box[9]) && ( (box[8] || box[2]) && (box[13] || box[12]) ) ){
					edges.data[(edges.cols*(j+1))+(i+1)]=0;
					edges.data[(edges.cols*(j-1))+(i+1)]=0;
					edges.data[(edges.cols*(j+1))+(i+2)]=0;
					edges.data[(edges.cols*(j-1))+(i+2)]=0;
					edges.data[(edges.cols*(j))+(i+1)]=255;
					edges.data[(edges.cols*(j))+(i+2)]=255;
				}

			}
		}

	//将连接的长且折的线段，断开。
	for(int j=start_y; j<end_y; j++)
		for(int i=start_x; i<end_x; i++){

			uchar box[33];
			/*
			点4为中心点
        	29		17         18        31
        		25	13         14  27
        	24	10	0    1     2   9     23
        			3    4     5
        	22	12	6    7     8   11    21
        		28  15         16  26
        	32		19         20        30

			*/

			box[4]=(uchar)edges.data[(edges.cols*(j))+(i)];

			if(box[4]){
				box[0]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box[1]=(uchar)edges.data[(edges.cols*(j-1))+(i)];
				box[2]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];

				box[3]=(uchar)edges.data[(edges.cols*(j))+(i-1)];
				box[5]=(uchar)edges.data[(edges.cols*(j))+(i+1)];

				box[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-1)];
				box[7]=(uchar)edges.data[(edges.cols*(j+1))+(i)];
				box[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];

				box[9]=(uchar)edges.data[(edges.cols*(j-1))+(i+2)];
				box[10]=(uchar)edges.data[(edges.cols*(j-1))+(i-2)];
				box[11]=(uchar)edges.data[(edges.cols*(j+1))+(i+2)];
				box[12]=(uchar)edges.data[(edges.cols*(j+1))+(i-2)];

				box[13]=(uchar)edges.data[(edges.cols*(j-2))+(i-1)];
				box[14]=(uchar)edges.data[(edges.cols*(j-2))+(i+1)];
				box[15]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box[16]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];

				box[17]=(uchar)edges.data[(edges.cols*(j-3))+(i-1)];
				box[18]=(uchar)edges.data[(edges.cols*(j-3))+(i+1)];
				box[19]=(uchar)edges.data[(edges.cols*(j+3))+(i-1)];
				box[20]=(uchar)edges.data[(edges.cols*(j+3))+(i+1)];

				box[21]=(uchar)edges.data[(edges.cols*(j+1))+(i+3)];
				box[22]=(uchar)edges.data[(edges.cols*(j+1))+(i-3)];
				box[23]=(uchar)edges.data[(edges.cols*(j-1))+(i+3)];
				box[24]=(uchar)edges.data[(edges.cols*(j-1))+(i-3)];

				box[25]=(uchar)edges.data[(edges.cols*(j-2))+(i-2)];
				box[26]=(uchar)edges.data[(edges.cols*(j+2))+(i+2)];
				box[27]=(uchar)edges.data[(edges.cols*(j-2))+(i+2)];
				box[28]=(uchar)edges.data[(edges.cols*(j+2))+(i-2)];

				box[29]=(uchar)edges.data[(edges.cols*(j-3))+(i-3)];
				box[30]=(uchar)edges.data[(edges.cols*(j+3))+(i+3)];
				box[31]=(uchar)edges.data[(edges.cols*(j-3))+(i+3)];
				box[32]=(uchar)edges.data[(edges.cols*(j+3))+(i-3)];

				//斜直线上，有中间两个点偏离到直线同一侧，则将该直线断开。将4号点从边缘中踢除。线段长度为4，拆成长度为1和2
				if( box[7] && box[2] && box[9] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[7] && box[0] && box[10] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[1] && box[8] && box[11] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[1] && box[6] && box[12] )
					edges.data[(edges.cols*(j))+(i)]=0;

				//断开倒了45度角的直角，将一个大直角线段（一条边垂直，一条边水平），拆成两条局部的直线（一条边垂直，一条边水平），原长度为7，拆成3和3
				if( box[0] && box[13] && box[17] && box[8] && box[11] && box[21] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[2] && box[14] && box[18] && box[6] && box[12] && box[22] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[6] && box[15] && box[19] && box[2] && box[9] && box[23] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[16] && box[20] && box[0] && box[10] && box[24] )
					edges.data[(edges.cols*(j))+(i)]=0;

				//将直角断开（一条边沿45度方向，一条边沿135度方向）	，长度为5，断开成2和2
				if( box[0] && box[25] && box[2] && box[27] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[0] && box[25] && box[6] && box[28] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[26] && box[2] && box[27] )
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box[8] && box[26] && box[6] && box[28] )
					edges.data[(edges.cols*(j))+(i)]=0;

				uchar box2[18];
				/*
		        3   16                5
		        	2	15        4   
		        		1   ij              
                    6   _   10    8
                7   _   13        11  9
		            14                12 
				*/
				box2[1]=(uchar)edges.data[(edges.cols*(j))+(i-1)];

				box2[2]=(uchar)edges.data[(edges.cols*(j-1))+(i-2)];
				box2[3]=(uchar)edges.data[(edges.cols*(j-2))+(i-3)];

				box2[4]=(uchar)edges.data[(edges.cols*(j-1))+(i+1)];
				box2[5]=(uchar)edges.data[(edges.cols*(j-2))+(i+2)];

				box2[6]=(uchar)edges.data[(edges.cols*(j+1))+(i-2)];
				box2[7]=(uchar)edges.data[(edges.cols*(j+2))+(i-3)];

				box2[8]=(uchar)edges.data[(edges.cols*(j+1))+(i+1)];
				box2[9]=(uchar)edges.data[(edges.cols*(j+2))+(i+2)];

				box2[10]=(uchar)edges.data[(edges.cols*(j+1))+(i)];

				box2[15]=(uchar)edges.data[(edges.cols*(j-1))+(i-1)];
				box2[16]=(uchar)edges.data[(edges.cols*(j-2))+(i-2)];

				box2[11]=(uchar)edges.data[(edges.cols*(j+2))+(i+1)];
				box2[12]=(uchar)edges.data[(edges.cols*(j+3))+(i+2)];

				box2[13]=(uchar)edges.data[(edges.cols*(j+2))+(i-1)];
				box2[14]=(uchar)edges.data[(edges.cols*(j+3))+(i-2)];

				if( box2[1] && box2[2] && box2[3] && box2[4] && box2[5] )  //将到了45度角的直角（一条边沿45度方向，另一条边沿着135度方向）断开。长度为6，拆成3和2
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[1] && box2[6] && box2[7] && box2[8] && box2[9] )  //同上，开口向下
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[10] && box2[11] && box2[12] && box2[4] && box2[5] )  //同上，开口向右
					edges.data[(edges.cols*(j))+(i)]=0;
				if( box2[10] && box2[13] && box2[14] && box2[15] && box2[16] )// 同上，开口向左。
					edges.data[(edges.cols*(j))+(i)]=0;
			}

		}
}

void PuRe::findPupilEdgeCandidates(const Mat &intensityImage, Mat &edge, vector<PupilCandidate> &candidates)
{
	/* Find all lines
	 * Small note here: using anchor points tends to result in better ellipse fitting later!
	 * It's also faster than doing connected components and collecting the labels
	 */
	vector<Vec4i> hierarchy;
	vector<vector<Point> > curves;

	//使用时，intensityImage为原图，edge为过滤后的边缘。

	// CV_RETR_LIST保留所有的contour到curves中，每个contour都是独立关系，没有嵌套。
	//CV_CHAIN_APPROX_TC89_KCOS，将一个离散封闭的图形边界，通过kcos识别关键性的角点，然后将这些角点串联成链，形成一个用多条直线近似描述的图形。
	//https://ieeexplore.ieee.org/document/31447
	//优点是减小了误差导致的部分偏移点，以及图像的离散坐标，对图形描述的精度。有点类似将位图用矢量直线表示
	findContours( edge, curves, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_KCOS );    //找轮廓其实就是在找连续或者封闭的边界点。
	removeDuplicates(curves, edge.cols);  //这个函数在头文件PuRe.h中实现了 功能为删除有交集的，或者重复的轮廓。如果两个轮廓只是有交集，这个函数应该还是有问题。
	if(0) {//调试  边缘轮廓，这一步效果不太明显
	    cv::Mat showContour = Mat::zeros(intensityImage.size(), CV_8UC3);
	    if(curves.size()>0){
            for (int i = 0;i< curves.size(); i++)
            {
                Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
                drawContours(showContour, curves, i, color, CV_FILLED, 8);
            }
            imshow("contours raw ", showContour);
	    } 
	}

    cv::Mat showContourCandidates = Mat::zeros(intensityImage.size(), CV_8UC3);
	// Create valid candidates
	for (size_t i=curves.size(); i-->0;) {
		PupilCandidate candidate(curves[i]);   //通过直线连接的轮廓来构建瞳孔的候选方案。
		if (candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias)){  //4个条标准，其中3个有指标进行检测， 这一步效果非常明显，用了瞳孔的先验知识，通常将虹膜也算进来了
			candidates.push_back(candidate );

	        if(0){ //调试  边缘轮廓候选 , isValid去掉了好多边缘，包括重要的边缘，有待改进
                Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
                drawContours(showContourCandidates, curves, i, color, CV_FILLED, 8);
                imshow("contours candidates ", showContourCandidates);
	        }
		}
	}
}
 

//
void PuRe::combineEdgeCandidates(const cv::Mat &intensityImage, cv::Mat &edge, std::vector<PupilCandidate> &candidates)  //相当于对已有的pupilCandidates，任取两个求并集，做成一个新的candidate，追加到已有的pupilCandidates集合里头。
{
	(void) edge;
	if (candidates.size() <= 1)
		return;
	vector<PupilCandidate> mergedCandidates;
	for (auto pc=candidates.begin(); pc!=candidates.end(); pc++) {
		for (auto pc2=pc+1; pc2!=candidates.end(); pc2++) {

			Rect intersection = pc->combinationRegion & pc2->combinationRegion;  //combinationRegion是包含CV_CHAIN_APPROX_TC89_KCOS生成的contour 边界点的最小正方形。
			if (intersection.area() < 1)
				continue; // no intersection
//#define DBG_EDGE_COMBINATION
#ifdef DBG_EDGE_COMBINATION
			Mat tmp;
			cvtColor(intensityImage, tmp, CV_GRAY2BGR);
			rectangle(tmp, pc->combinationRegion, pc->color);
			for (unsigned int i=0; i<pc->points.size(); i++)
				cv::circle(tmp, pc->points[i], 1, pc->color, -1);
			rectangle(tmp, pc2->combinationRegion, pc2->color);
			for (unsigned int i=0; i<pc2->points.size(); i++)
				cv::circle(tmp, pc2->points[i], 1, pc2->color, -1);
			imshow("Combined edges", tmp);
			imwrite("combined.png", tmp);
			//waitKey(0);
#endif

			if (intersection.area() >= min<int>(pc->combinationRegion.area(),pc2->combinationRegion.area()))  //如果是一个区域包含另一个区域，就跳过
				continue;

			vector<Point> mergedPoints = pc->points;
			mergedPoints.insert(mergedPoints.end(), pc2->points.begin(), pc2->points.end());   //把两个有相交部分的轮廓上的点，直接都连起来，构成一个新的轮廓。
			PupilCandidate candidate( mergedPoints );
			if (!candidate.isValid(intensityImage, minPupilDiameterPx, maxPupilDiameterPx, outlineBias))   //合成后的4条原则，要满足
				continue;
			if (candidate.outlineContrast < pc->outlineContrast || candidate.outlineContrast < pc2->outlineContrast)  //合成后的边界对比度，要比不合成的大
				continue;
			mergedCandidates.push_back( candidate );
		}
	}
	candidates.insert( candidates.end(), mergedCandidates.begin(), mergedCandidates.end() );    //将合成的candidate 追加到原来的candidate里头。
}

//给定一个candidate，找到主轴小于它，且边缘对比度大于0.75，且中心距离小于搜索距离的所有轮廓。
//根据得分score排序，函数实现见PuRe.h中的 < 的重载。选择得分最大的。
void PuRe::searchInnerCandidates(vector<PupilCandidate> &candidates, PupilCandidate &candidate)  
{
	if (candidates.size() <= 1)
		return;

	float searchRadius = 0.5*candidate.majorAxis;
	vector<PupilCandidate> insiders;
	for (auto pc=candidates.begin(); pc!=candidates.end(); pc++) {       
		if (searchRadius < pc->majorAxis)
			continue;
		if (norm( candidate.outline.center - pc->outline.center) > searchRadius)
			continue;
		if (pc->outlineContrast < 0.75)
			continue;
		insiders.push_back(*pc);
	}
	if (insiders.size() <= 0) {
		//ellipse(dbg, candidate.outline, Scalar(0,255,0));
		return;
	}

	sort( insiders.begin(), insiders.end() );  //根据得分score排序，函数实现见PuRe.h中的 < 的重载。
	candidate = insiders.back(); // 选择得分最大的。

	//circle(dbg, searchCenter, searchRadius, Scalar(0,0,255),3);
	//candidate.draw(dbg);
	//imshow("dbg", dbg);
}

void PuRe::detect(Pupil &pupil, const cv::Mat& rawFrame)
{
	// 3.2 Edge Detection and Morphological Transformation
	Mat detectedEdges = canny(input, true, true, 64, 0.7f, 0.4f);

	//imshow("edges raw", detectedEdges);
#ifdef SAVE_ILLUSTRATION
	imwrite("edges.png", detectedEdges);
#endif

	filterEdges(detectedEdges);
	//imshow("edges filtered", detectedEdges);
	// 3.3 Segment Selection
	vector<PupilCandidate> candidates;
	findPupilEdgeCandidates(input, detectedEdges, candidates);   
	if (candidates.size() <= 0){
		currentPupilConfidence=0;
		return;
	}
		

	//for ( auto c = candidates.begin(); c != candidates.end(); c++)
	//	c->draw(dbg);

#ifdef SAVE_ILLUSTRATION
	float r = 255.0 / candidates.size();
	int i = 0;
	Mat candidatesImage;
	cvtColor(input, candidatesImage, CV_GRAY2BGR);
	for ( auto c = candidates.begin(); c != candidates.end(); c++) {
		Mat colorMat = (Mat_<uchar>(1,1) << i*r);
		applyColorMap(colorMat, colorMat, COLORMAP_HSV);
		c->color = colorMat.at<Vec3b>(0,0);
		c->draw(candidatesImage, c->color );
		i++;
	}
	imwrite ("input.png", input);
	imwrite ("filtered-edges.png", detectedEdges);
	imwrite("candidates.png", candidatesImage);
#endif

	// Combination
	combineEdgeCandidates(input, detectedEdges, candidates);  //这一步容易将错误的边界纳入进来。例如虹膜上的纹理。容易瞳孔边缘链接出错，形成内部链接
	for (auto c=candidates.begin(); c!=candidates.end(); c++) {
		if (c->outlineContrast < 0.5)
			c->score = 0;
		if (c->outline.size.area() > CV_PI*pow(0.5*maxPupilDiameterPx,2))
			c->score = 0;
		if (c->outline.size.area() < CV_PI*pow(0.5*minPupilDiameterPx,2))
			c->score = 0;
	}

	/*
	for ( int i=0; i<candidates.size(); i++) {
		Mat out;
		cvtColor(input, out, CV_GRAY2BGR);
		auto c = candidates[i];
		c.drawit(out, c.color);
		imwrite(QString("candidate-%1.png").arg(i).toStdString(), out);
		c.drawOutlineContrast(input, 5, QString("contrast-%1-%2.png").arg(i).arg(QString::number(c.score)));
		//waitKey(0);
	}
	*/

	// Scoring
	sort( candidates.begin(), candidates.end() );
	PupilCandidate selected = candidates.back();  //找到得分最大的candidate

	//for ( auto c = candidates.begin(); c != candidates.end(); c++)
	//    c->draw(dbg);

	if(0) {//调试  边缘轮廓，这一步效果不太明显，主要是前面的findPupilEdgeCandidates中的isValid判断（根据瞳孔先验知识判断，效果明显）
	    cv::Mat showContour = Mat::zeros(input.size(), CV_8UC3);
		std::vector<std::vector<cv::Point> > combinedContours;
	    if(candidates.size()>0){
			for (int i = 0;i< candidates.size(); i++){
				combinedContours.push_back(candidates[i].points);
            }
            for (int i = 0;i< candidates.size(); i++){
                Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
                drawContours(showContour, combinedContours, i, color, CV_FILLED, 8);
            }
            imshow("combined contours ", showContour);
	    } 
	}
	// Post processing
	searchInnerCandidates(candidates, selected);  //找到已识别的candidate里头是否有更好的，将结果保存到selected里头

	//pupil=selected.outline;
	pupil.confidence = selected.outlineContrast;
	currentPupilConfidence=pupil.confidence;
    
	//在PuRe基础上新加的功能。
	referenceInitalizedFrames++;
	refine_pupil_and_find_glints(rawFrame,input,selected,detectedEdges,5);   
	pupil = pupilOutline;

#ifdef SAVE_ILLUSTRATION
	Mat out;
	cvtColor(input, out, CV_GRAY2BGR);
	ellipse(out, pupil, Scalar(0,255,0), 2);
	line(out, Point(pupil.center.x,0), Point(pupil.center.x,out.rows), Scalar(0,255,0), 2);
	line(out, Point(0,pupil.center.y), Point(out.cols,pupil.center.y), Scalar(0,255,0), 2);
	imwrite("out.png", out);
#endif
}

void PuRe::run(const Mat &frame, Pupil &pupil)            //头文件中定义的run，调用这个run
{
	pupil.clear();
	init(frame);  //根据baseSize计算要对图像进行放缩的比例。

	// Downscaling
	Mat downscaled;
	resize(frame, downscaled, Size(), scalingRatio, scalingRatio, CV_INTER_LINEAR);  
	normalize(downscaled, input, 0, 255, NORM_MINMAX, CV_8U);  //将数据规范化到0-255；

	workingSize.width = floor(scalingRatio*frame.cols);
	workingSize.height = floor(scalingRatio*frame.rows);

	// Estimate parameters based on the working size
	estimateParameters(workingSize.height, workingSize.width);  //估算瞳孔的先验参数，先验假设中，假设了左右眼角的距离占整个图像的1/5-1/2。相当于提前设置了瞳孔的像素直径范围。

	// Preallocate stuff for edge detection
	dx = Mat::zeros(workingSize, CV_32F);
	dy = Mat::zeros(workingSize, CV_32F);
	magnitude = Mat::zeros(workingSize, CV_32F);
	edgeType = Mat::zeros(workingSize, CV_8U);
	edge = Mat::zeros(workingSize, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*minPupilDiameterPx, Scalar(0,0,0), 2);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*maxPupilDiameterPx, Scalar(0,0,0), 3);

	// Detection
	detect(pupil,frame);
	if(currentPupilConfidence>0){
	    pupil.resize( 1.0 / scalingRatio, 1.0 / scalingRatio );  //返回到原来图像中的瞳孔位姿
	    if(leftGilntPositionCurrent.x>-9000)
	        leftGilntPositionCurrentInRawImage =leftGilntPositionCurrent; //还原到原始图中的glint坐标
	    else{
	    	leftGilntPositionCurrentInRawImage.x=-10000;
	    	leftGilntPositionCurrentInRawImage.y=-10000;
	    }
    
	    
        if(rightGilntPositionCurrent.x>-9000)
	        rightGilntPositionCurrentInRawImage=rightGilntPositionCurrent;
	    else{
	    	rightGilntPositionCurrentInRawImage.x=-10000;
	    	rightGilntPositionCurrentInRawImage.y=-10000;
	    }
	}
	//imshow("dbg", dbg);
}

void PuRe::run(const cv::Mat &frame, const cv::Rect &roi, Pupil &pupil, const float &userMinPupilDiameterPx, const float &userMaxPupilDiameterPx)  //这个run可以自定义roi，速度可以加快。
{
	if (roi.area() < 10) {
		//qWarning() << "Bad ROI: falling back to regular detection.";
		run(frame, pupil);
		return;
	}

	pupil.clear();

	init(frame);

	estimateParameters(scalingRatio*frame.rows, scalingRatio*frame.cols);
	if (userMinPupilDiameterPx > 0)
		minPupilDiameterPx = scalingRatio*userMinPupilDiameterPx;
	if (userMaxPupilDiameterPx > 0)
		maxPupilDiameterPx = scalingRatio*userMaxPupilDiameterPx;

	// Downscaling
	Mat downscaled;
	resize(frame(roi), downscaled, Size(), scalingRatio, scalingRatio, CV_INTER_LINEAR);
	normalize(downscaled, input, 0, 255, NORM_MINMAX, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);

	workingSize.width = input.cols;
	workingSize.height = input.rows;

	// Preallocate stuff for edge detection
	dx = Mat::zeros(workingSize, CV_32F);
	dy = Mat::zeros(workingSize, CV_32F);
	magnitude = Mat::zeros(workingSize, CV_32F);
	edgeType = Mat::zeros(workingSize, CV_8U);
	edge = Mat::zeros(workingSize, CV_8U);

	//cvtColor(input, dbg, CV_GRAY2BGR);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*minPupilDiameterPx, Scalar(0,0,0), 2);
	//circle(dbg, Point(0.5*dbg.cols,0.5*dbg.rows), 0.5*maxPupilDiameterPx, Scalar(0,0,0), 3);

	// Detection
	detect(pupil,frame);

	if(currentPupilConfidence>0){
		pupil.resize( 1.0 / scalingRatio, 1.0 / scalingRatio );
	    pupil.center += Point2f(roi.tl());
	}
	//imshow("dbg", dbg);
}

/*******************************************************************************
 *
 * Pupil Candidate Functions
 *
 ******************************************************************************/
//每一度，一个数。
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


/*
栈空间是有限的，假如频繁大量的使用就会造成因栈空间不足而导致程序出错的问题，如，函数的死循环递归调用的最终结果就是导致栈内存空间枯竭
inline 的使用是有所限制的，inline 只适合涵数体内代码简单的涵数使用，不能包含复杂的结构控制语句例如 while、switch，并且不能内联函数本身不能是直接递归函数（即，自己内部还调用自己的函数）。
*/


static void inline sincos( int angle, float& cosval, float& sinval )  //通过查表快速得到sin和cos
{
	angle += (angle < 0 ? 360 : 0);
	sinval = sinTable[angle];
	cosval = sinTable[450 - angle];
}

static inline vector<Point> ellipse2Points( const RotatedRect &ellipse, const int &delta )  //椭圆离散化成点
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

inline bool PupilCandidate::isValid(const cv::Mat &intensityImage, const int &minPupilDiameterPx, const int &maxPupilDiameterPx, const int bias)
{
	if (points.size() < 5)   //至少要5个点以上的，才能算作瞳孔
		return false;

	float maxGap = 0;
	for (auto p1=points.begin(); p1!=points.end(); p1++) {    //计算轮廓上最大的两个点的距离。
		for (auto p2=p1+1; p2!=points.end(); p2++) {
			float gap = norm(*p2-*p1);
			if (gap > maxGap)
				maxGap = gap;
		}
	}

	if ( maxGap >= maxPupilDiameterPx )     //根据轮廓上的任意两点间最大距离和最小距离的约束，去掉瞳孔直径以外的轮廓。
		return false;
	if ( maxGap <= minPupilDiameterPx )
		return false;

	outline = fitEllipse(points);        //直接用CV_CHAIN_APPROX_TC89_KCOS近似勾画的轮廓产生的直线的角点拟合椭圆，这样一定程度上，缓解了拟合时拟合样本点分布不均匀的问题。
	boundaries = {0, 0, intensityImage.cols, intensityImage.rows};

	if (!boundaries.contains(outline.center))      //整幅图要包含椭圆的中心
		return false;

	if (!fastValidityCheck(maxPupilDiameterPx) )   //1、检验拟合后的椭圆的长短轴比和长轴的像素长度，生成一个未旋转的正方形combinationRegion
		return false;

	pointsMinAreaRect = minAreaRect(points);  //2、找到一个旋转的最小矩形来包含轮廓点
	if (ratio(pointsMinAreaRect.size.width,pointsMinAreaRect.size.height) < minCurvatureRatio) //轮廓的实际长短轴比值检验
		return false;

	if (!validityCheck(intensityImage, bias))  //3、每十度一步，将椭圆离散成点，沿着椭圆中心和边界点，形成一条直线，沿着这条线，往边界点内外偏移，得到的两个点，这两个点分别和边界点连成线段，两条线段上的亮度平均值，要大于bias。
		return false;

	updateScore();   //平均了前面三项有效性检测的得分，并赋值给score
	return true;
}

inline bool PupilCandidate::fastValidityCheck(const int &maxPupilDiameterPx)//对拟合后的椭圆进行长短轴检验。
{
	pair<float,float> axis = minmax(outline.size.width, outline.size.height);
	minorAxis = axis.first;
	majorAxis = axis.second;
	aspectRatio = minorAxis / majorAxis;

	if (aspectRatio < minCurvatureRatio)
		return false;

	if (majorAxis > maxPupilDiameterPx)
		return false;

	combinationRegion = boundingRect(points);  //找一个包含轮廓的正方形
	combinationRegion.width = max<int>(combinationRegion.width, combinationRegion.height);
	combinationRegion.height = combinationRegion.width;

	return true;
}


//每十度一步，将椭圆离散成点，沿着椭圆中心和边界点，形成一条直线，沿着这条线，往边界点内外偏移，得到的两个点，这两个点分别和边界点连成线段，两条线段上的亮度平均值，要大于bias。这种点的比例，就是轮廓对比度的有效性计算。
inline bool PupilCandidate::validateOutlineContrast(const Mat &intensityImage, const int &bias) 
{
	int delta = 0.15*minorAxis;
	cv::Point c = outline.center;
//#define DBG_OUTLINE_CONTRAST
#ifdef DBG_OUTLINE_CONTRAST
	cv::Mat tmp;
	cv::cvtColor(intensityImage, tmp, CV_GRAY2BGR);
	cv::ellipse(tmp, outline, cv::Scalar(0,255,255));
	cv::Scalar lineColor;
#endif
	int evaluated = 0;
	int validCount = 0;

	vector<Point> outlinePoints = ellipse2Points(outline, 10);   //每十度一步，将椭圆离散成点，沿着椭圆中心和边界点
	for (auto p=outlinePoints.begin(); p!=outlinePoints.end(); p++) {
		int dx = p->x - c.x;
		int dy = p->y - c.y;

		float a = 0;
		if (dx != 0)
			a = dy / (float) dx;
		float b = c.y - a*c.x;

		if (a == 0)
			continue;

		if ( abs(dx) > abs(dy) ) {
			int sx = p->x - delta;
			int ex = p->x + delta;
			int sy = std::roundf(a*sx + b);
			int ey = std::roundf(a*ex + b);
			cv::Point start = { sx, sy };
			cv::Point end = { ex, ey };
			evaluated++;

			if (!boundaries.contains(start) || !boundaries.contains(end) )
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int x=sx; x<p->x; x++)
				m1 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int x=p->x+1; x<=ex; x++) {
				m2 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			}
			m2 = std::roundf( m2 / delta );

#ifdef DBG_OUTLINE_CONTRAST
			lineColor = cv::Scalar(0,0,255);
#endif
			if (p->x < c.x) {// leftwise point
				if (m1 > m2+bias) {
					validCount ++;
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
		} else {
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
				m1 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int y=p->y+1; y<=ey; y++)
				m2 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
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
		return false;
	outlineContrast = validCount / (float) evaluated;
#ifdef DBG_OUTLINE_CONTRAST
	cv::imshow("Outline Contrast", tmp);
	cv::waitKey(0);
#endif

	return true;
}


//CV_CHAIN_APPROX_TC89_KCOS生成的用于近似原边缘的直线串组合points。判断这些points是否围绕outline四象限都有分布，只要每个象限有一个点就行，整个针对于CV_CHAIN_APPROX_TC89_KCOS输出的矢量边界点，很合理。
inline bool PupilCandidate::validateAnchorDistribution()  
{
	anchorPointSlices.reset();
	for (auto p=points.begin(); p!=points.end(); p++) {
		if (p->x - outline.center.x < 0) {
			if (p->y - outline.center.y < 0)
				anchorPointSlices.set(Q0);
			else
				anchorPointSlices.set(Q3);
		} else  {
			if (p->y - outline.center.y < 0)
				anchorPointSlices.set(Q1);
			else
				anchorPointSlices.set(Q2);
		}
	}
	anchorDistribution = anchorPointSlices.count() / (float) anchorPointSlices.size();
	return true;
}


inline bool PupilCandidate:: validityCheck(const cv::Mat &intensityImage, const int &bias)  
{
	mp = std::accumulate(points.begin(), points.end(), cv::Point(0,0) );  ////CV_CHAIN_APPROX_TC89_KCOS生成的用于近似原边缘的直线串组合。轮廓的几何中心
	mp.x = std::roundf(mp.x / points.size());
	mp.y = std::roundf(mp.y / points.size());

	outline.points(v);           //拟合后的椭圆的外截四边形的4个顶点。
	std::vector<cv::Point2f> pv(v, v+sizeof(v)/sizeof(v[0]));
	if (cv::pointPolygonTest(pv, mp, false) <= 0)  //拟合后的椭圆要包含几何中心，这个应该时多余的一步。
		return false;

	if (!validateAnchorDistribution())   //CV_CHAIN_APPROX_TC89_KCOS生成的用于近似原边缘的直线串组合points。判断这些points是否围绕outline四象限都有分布，只要每个象限有一个点就行，这个不一定合适。
		return false;

	if (!validateOutlineContrast(intensityImage, bias))  //每十度一步，将椭圆离散成点，沿着椭圆中心和边界点，形成一条直线，沿着这条线，往边界点内外偏移，得到的两个点，这两个点分别和边界点连成线段，两条线段上的亮度平均值，要大于bias。
		return false;

	return true;
}


inline bool PupilCandidate::drawOutlineContrast(const Mat &intensityImage, const int &bias, string out)  //这个函数是用来调试显示数据的，目前没有用
{
	double lw = 1;
	int delta = 0.15*minorAxis;
	cv::Point c = outline.center;
	cv::Mat tmp;
	cv::cvtColor(intensityImage, tmp, CV_GRAY2BGR);
	cv::ellipse(tmp, outline, cv::Scalar(0,255,255), lw);
	cv::Scalar lineColor;

	int evaluated = 0;
	int validCount = 0;


	vector<Point> outlinePoints = ellipse2Points(outline, 10);
	for (auto p=outlinePoints.begin(); p!=outlinePoints.end(); p++) {
		int dx = p->x - c.x;
		int dy = p->y - c.y;

		float a = 0;
		if (dx != 0)
			a = dy / (float) dx;
		float b = c.y - a*c.x;

		if (a == 0)
			continue;

		if ( abs(dx) > abs(dy) ) {
			int sx = p->x - delta;
			int ex = p->x + delta;
			int sy = std::roundf(a*sx + b);
			int ey = std::roundf(a*ex + b);
			cv::Point start = { sx, sy };
			cv::Point end = { ex, ey };
			evaluated++;

			if (!boundaries.contains(start) || !boundaries.contains(end) )
				continue;

			float m1, m2, count;

			m1 = count = 0;
			for (int x=sx; x<p->x; x++)
				m1 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int x=p->x+1; x<=ex; x++) {
				m2 += intensityImage.ptr<uchar>( (int) std::roundf(a*x+b) )[x];
			}
			m2 = std::roundf( m2 / delta );

			lineColor = cv::Scalar(0,0,255);
			if (p->x < c.x) {// leftwise point
				if (m1 > m2+bias) {
					validCount ++;
					lineColor = cv::Scalar(0,255,0);
				}
			} else {// rightwise point
				if (m2 > m1+bias) {
					validCount++;
					lineColor = cv::Scalar(0,255,0);
				}
			}

			cv::line(tmp, start, end, lineColor, lw);
		} else {
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
				m1 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m1 = std::roundf( m1 / delta );

			m2 = count = 0;
			for (int y=p->y+1; y<=ey; y++)
				m2 += intensityImage.ptr<uchar>(y)[ (int) std::roundf((y-b)/a) ];
			m2 = std::roundf( m2 / delta );

			lineColor = cv::Scalar(0,0,255);
			if (p->y < c.y) {// upperwise point
				if (m1 > m2+bias) {
					validCount++;
					lineColor = cv::Scalar(0,255,0);
				}
			} else {// bottomwise point
				if (m2 > m1+bias) {
					validCount++;
					lineColor = cv::Scalar(0,255,0);
				}
			}

			cv::line(tmp, start, end, lineColor, lw);
		}
	}
	if (evaluated == 0)
		return false;
	outlineContrast = validCount / (float) evaluated;
	cv::imwrite(out, tmp);

	return true;
}

/*此函数在detect的结果的基础上，进一步对瞳孔边缘，glint反光斑，进行识别。
input： resize后的单通道原始图像。
selected： PuRe::detect函数输出的初步结果。其中的边缘点很稀疏。
detectedEdges： PuRe::filterEdges得到的边缘，例如去掉了邻居较多的边缘点。边缘点很稠密。
contrastStep： 判断瞳孔边缘时，沿着边缘梯度方向，前后两侧取平均灰度的步长，以求边缘两侧的对比度。
*/
void PuRe::refine_pupil_and_find_glints(const cv::Mat& rawFrame, cv::Mat& input,PupilCandidate& selected, cv::Mat& detectedEdges, int contrastStep)
{
	leftGilntPositionCurrent.x=-10000;
	leftGilntPositionCurrent.y=-10000;
	rightGilntPositionCurrent.x=-10000;
	rightGilntPositionCurrent.y=-10000;
	pupilOutline.center.x=-10000;
	pupilOutline.center.y=-10000;

    int pupilEdgeContrastStep=10;

	//先对瞳孔预估位置，按照内黑/睫毛，虹膜灰
	cv::Rect pupilRoi = boundingRect(selected.points);  
	double pupilExpandRatio = 2; //在终选的瞳孔的拓展范围内进行边界查找。

	if(1){	
        cv::imshow("raw detected edge", detectedEdges);
	}

    /* 第一步： 将断掉的瞳孔边缘纳入进来，对已经选中的边缘，统计边缘的特征，包括梯度前后的对比度，亮侧平均值，暗侧平局值。
	求出对比度较高的70%点的均值，标准差，用来忽略睫毛的影响（睫毛边缘的对比度较低）
	求出暗侧亮度较低的70%点的均值，标准差，用来忽略反光斑的影响（反光斑暗侧的亮度较高）
	然后通过贪婪法找PuRe输出结果相连的，切满足条件的点。
	*/
	int PuReEdgeLength = selected.points.size();
	float pupilEdgeContrast;   //用来保存类外函数的返回值
	float pupilEdgeLighterSideIntensity; //同上
	float pupilEdgeDarkerSideIntensity; //同上
	int pupilEdgeContrastHist[256]={0};   //边缘特征的直方图
	int pupilEdgeLighterSideIntensityHist[256]={0}; 
	int pupilEdgeDarkerSideIntensityHist[256]={0};
	std::vector<t_edgeFeatures> edgesFeatures; 
	t_edgeFeatures edgeFeature;
	std::vector<cv::Point> pupilEdgeConnectionInit;

	//筛选出边缘特征能够求出来的边缘点，并保存边缘特征。
	for(int i=0;i<PuReEdgeLength;i++){
		if(contrast_along_gradient(selected.points.at(i), input, dx, dy, pupilEdgeContrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
		    //printf("point(%d,%d) edgeContrast: %f; lighter side intensity: %f; darker side intensity: %f\n",selected.points.at(i).x,selected.points.at(i).y,pupilEdgeContrast,pupilEdgeLighterSideIntensity,pupilEdgeDarkerSideIntensity);
			pupilEdgeConnectionInit.push_back(selected.points.at(i));
			pupilEdgeContrastHist[(int)pupilEdgeContrast]++;
		    pupilEdgeLighterSideIntensityHist[(int)pupilEdgeLighterSideIntensity]++;
		    pupilEdgeDarkerSideIntensityHist[(int)pupilEdgeDarkerSideIntensity]++;

			edgeFeature.x = selected.points.at(i).x;
			edgeFeature.y = selected.points.at(i).y;
			edgeFeature.contrast = pupilEdgeContrast;
			edgeFeature.darkerSideIntensity = pupilEdgeDarkerSideIntensity;
			edgeFeature.lighterSideIntensity = pupilEdgeLighterSideIntensity;
			edgesFeatures.push_back(edgeFeature);
		}
	}
	//以边缘暗侧的平均亮度的直方图中，较低的80%点的最大亮度值darkerSideThre。主要用来去掉反光斑边缘的影响。
	float darkerSideThre;
	float lightSideThre;
	float contrastThre;
	int darkerSideThreNum=0;
	int lightSideThreNum=0;
	int contrastThreNum=0;
	for(int i=0;i<256;i++){
		if(darkerSideThreNum < 0.8*edgesFeatures.size()){
		    darkerSideThreNum += pupilEdgeDarkerSideIntensityHist[i];
			darkerSideThre = i;
			//printf("darker side intensity value: %d; number: %d; points number: %d; points needed: %d\n",i,pupilEdgeDarkerSideIntensityHist[i],darkerSideThreNum,(int)(0.7*edgesFeatures.size()));
		}
		else
            break;
	}
	//以边缘亮侧的平均亮度的直方图中，较低的80%点的最大亮度值lightSideThre。主要用来去掉反光斑边缘的影响。
	for(int i=0;i<256;i++){
		if(lightSideThreNum < 0.8*edgesFeatures.size()){
		    lightSideThreNum += pupilEdgeLighterSideIntensityHist[i];
			lightSideThre = i;
			//printf("darker side intensity value: %d; number: %d; points number: %d; points needed: %d\n",i,pupilEdgeDarkerSideIntensityHist[i],darkerSideThreNum,(int)(0.7*edgesFeatures.size()));
		}
		else
            break;
	}
	//以边缘对比度的平均对比度的直方图中，较高的80%点中的最小对比度值contrastThre。主要用来去掉睫毛的影响。
	for(int i=0;i<256;i++){
		if(contrastThreNum < 0.2*edgesFeatures.size()){
			contrastThreNum+= pupilEdgeContrastHist[i];
			contrastThre=i;
			//printf("contrast value: %d; number: %d; points number: %d; points needed: %d\n",i,pupilEdgeContrastHist[i],contrastThreNum,(int)(0.3*edgesFeatures.size()));
		}
		else
            break;
	}
	//printf("darker side threshold: %f; light side threshold: %f; edge contrast threshold: %f; total PuRe points： %d\n",darkerSideThre,lightSideThre,contrastThre,PuReEdgeLength);
	//在边缘中寻找同时满足以上两个条件，暗侧亮度值小于darkerSideThre，对比度大于contrastThre的点。即为去掉了睫毛和反光斑的影响。
	//求出暗侧亮度值 mu 和标准差 sd，然后以 mu+10sd 为暗侧的亮度的上限阈值。
	//求出对比度平局值 mu和sd，然后以 mu-10sd 为对比度的下限阈值。
	//求出亮侧亮度值 mu 和标准差 sd，然后以 mu+10sd 为亮侧的亮度的上限阈值。
	float edgeContrastMu=0;
	float darkerSideIntensityMu=0;
	float lighterSideIntensityMu=0;
	float edgeContrastSD=0;
	float darkerSideIntensitySD=0;
	float lighterSideIntensitySD=0;
	int validPupilEdgePointsNum=0;
	//求均值
	for(int i=0;i<edgesFeatures.size();i++){
		if(edgesFeatures[i].contrast>contrastThre && edgesFeatures[i].darkerSideIntensity<darkerSideThre){
			edgeContrastMu+=edgesFeatures[i].contrast;
			darkerSideIntensityMu+=edgesFeatures[i].darkerSideIntensity;
			lighterSideIntensityMu+=edgesFeatures[i].lighterSideIntensity;
			validPupilEdgePointsNum++;
		}
	}
	edgeContrastMu/=validPupilEdgePointsNum;
	darkerSideIntensityMu/=validPupilEdgePointsNum;
	lighterSideIntensityMu/=validPupilEdgePointsNum;

	//求标准差
	for(int i=0;i<edgesFeatures.size();i++){
		if(edgesFeatures[i].contrast>contrastThre && edgesFeatures[i].darkerSideIntensity<darkerSideThre){
			edgeContrastSD += pow((edgesFeatures[i].contrast-edgeContrastMu),2);
			darkerSideIntensitySD += pow((edgesFeatures[i].darkerSideIntensity-darkerSideIntensityMu),2);
			lighterSideIntensitySD+=pow((edgesFeatures[i].lighterSideIntensity-lighterSideIntensityMu),2);
		}
	}
	edgeContrastSD = sqrt(edgeContrastSD/validPupilEdgePointsNum);
	darkerSideIntensitySD = sqrt(darkerSideIntensitySD/validPupilEdgePointsNum);
	lighterSideIntensitySD= sqrt(lighterSideIntensitySD/validPupilEdgePointsNum);

	//以PuRe输出的特征点作为种子点，将与种子点向连的满足条件的点，纳入到瞳孔边界中。利用已经识别的瞳孔边缘的某点的梯度方向，链接有细微断开的边缘，所在方位和梯度方向夹角大于一定角度，例如arccos(0.3)。
	float darkerSideIntensityHighLimit;
	float lighterSideIntensityHighLimit;
	float edgeContrastLowLimit;
	float errorBandWidth=3;
	darkerSideIntensityHighLimit= (darkerSideIntensityMu + errorBandWidth*darkerSideIntensitySD)<128? (darkerSideIntensityMu + errorBandWidth*darkerSideIntensitySD):128;
	lighterSideIntensityHighLimit = (lighterSideIntensityMu+ errorBandWidth*lighterSideIntensitySD)<200?(lighterSideIntensityMu+ errorBandWidth*lighterSideIntensitySD):200;
    edgeContrastLowLimit = (edgeContrastMu-errorBandWidth*edgeContrastSD)<10?10:(edgeContrastMu-errorBandWidth*edgeContrastSD);  //新相机的图像偏暗，最低对比度偏低。

	std::vector<cv::Point> pupilEdgeWithMoreEdges;
	// for(int i=0;i<pupilEdgeConnectionInit.size();i++){
	// 	recursive_greedy_connectivity_search(pupilEdgeWithMoreEdges,pupilEdgeConnectionInit.at(i),detectedEdges,pupilEdgeContrastStep,darkerSideIntensityHighLimit,lighterSideIntensityHighLimit,edgeContrastLowLimit);
	// }

	direct_greedy_connectivity_search(pupilEdgeWithMoreEdges,pupilEdgeConnectionInit,detectedEdges,pupilEdgeContrastStep, darkerSideIntensityHighLimit,lighterSideIntensityHighLimit,edgeContrastLowLimit);

	//显示纳入断开的瞳孔边缘点
	cv::RotatedRect ellipseFitted;
	cv::Rect correctedPupilRoi(0,0,0,0);
	cv::Rect correctedExpandedPupilRoi(0,0,0,0);

	if(pupilEdgeWithMoreEdges.size()>5){
		ellipseFitted = cv::fitEllipse(pupilEdgeWithMoreEdges);
		pupilOutline = ellipseFitted;
		cv::Rect temp1 = ellipseFitted.boundingRect();        
		cv::Rect temp2= cv::boundingRect(pupilEdgeWithMoreEdges);
		int ellipseCenterX=temp1.x+temp1.width/2;
		int ellipseCenterY=temp1.y+temp1.height/2;
		if(ellipseCenterX > temp2.x && ellipseCenterX<(temp2.x+temp2.width) && ellipseCenterY >temp2.y && ellipseCenterY<(temp2.y+temp2.height)){
		    correctedPupilRoi = temp1; 
			correctedExpandedPupilRoi=cv::Rect(correctedPupilRoi.x-(pupilExpandRatio-1)/2*correctedPupilRoi.width,correctedPupilRoi.y-(pupilExpandRatio-1)/2*correctedPupilRoi.height,pupilExpandRatio*correctedPupilRoi.width,pupilExpandRatio*correctedPupilRoi.height);
			correctedExpandedPupilRoi.x = correctedExpandedPupilRoi.x<0?0:correctedExpandedPupilRoi.x;
			correctedExpandedPupilRoi.y = correctedExpandedPupilRoi.y<0?0:correctedExpandedPupilRoi.y;
	        correctedExpandedPupilRoi.width = (correctedExpandedPupilRoi.x+correctedExpandedPupilRoi.width)<=input.cols ? correctedExpandedPupilRoi.width : (input.cols -correctedExpandedPupilRoi.x);
	        correctedExpandedPupilRoi.height = (correctedExpandedPupilRoi.y+correctedExpandedPupilRoi.height)<=input.rows ? correctedExpandedPupilRoi.height : (input.rows -correctedExpandedPupilRoi.y);
			//printf("correctedExpandedPupilRoi(x,y,width,heigth):(%d,%d,%d,%d)\n",correctedExpandedPupilRoi.x,correctedExpandedPupilRoi.y,correctedExpandedPupilRoi.width,correctedExpandedPupilRoi.height);
		}

		//printf("ellipseFitted.size:(%f,%f); maxPupilSizeDuringTwoGlint:%f\n",ellipseFitted.size.height,ellipseFitted.size.width,maxPupilSizeDuringTwoGlint);
		if(referenceInitializedForBothGlintsFlag){
			if(ellipseFitted.size.height > maxPupilSizeDuringTwoGlint*1.05 || ellipseFitted.size.width > maxPupilSizeDuringTwoGlint*1.05 ||
			   (pupilOutline.center.x - pupilOutline.size.width/2)<0 || (pupilOutline.center.x + pupilOutline.size.width/2)>=detectedEdges.cols ||
			   (pupilOutline.center.y - pupilOutline.size.height/2)<0 || (pupilOutline.center.y + pupilOutline.size.height/2)>=detectedEdges.rows || 
			    pupilOutline.size.width<0 || 
				pupilOutline.size.height<0){
				correctedExpandedPupilRoi.width=0;
				correctedExpandedPupilRoi.height=0;
				currentPupilConfidence=0;
			}
		}
	}
	else{
		pupilOutline = cv::RotatedRect(cv::Point2f(1,1),cv::Size2f(1,1),0);
	}

	if(1){
		cv::Mat rawSelectedPupilEdges = Mat::zeros(input.size(), CV_8UC1);  
		rawSelectedPupilEdges.setTo(0);
	    for(int i=0;i<selected.points.size();i++){  
	    	rawSelectedPupilEdges.ptr<uchar>(selected.points.at(i).y)[selected.points.at(i).x]=255;
	    }
	    cv::imshow("raw selected pupil edges", rawSelectedPupilEdges);

	    cv::Mat moreConnectedPupilEdges = Mat::zeros(input.size(), CV_8UC1);  
	    moreConnectedPupilEdges.setTo(0);
        for(int i=0;i<pupilEdgeWithMoreEdges.size();i++){
        out23DEyeTracker.emplace_back(cv::Point2f(pupilEdgeWithMoreEdges.at(i).x,pupilEdgeWithMoreEdges.at(i).y));
        }
	    for(int i=0;i<pupilEdgeWithMoreEdges.size();i++){  
	    	moreConnectedPupilEdges.ptr<uchar>(pupilEdgeWithMoreEdges.at(i).y)[pupilEdgeWithMoreEdges.at(i).x]=255;
	    }
	    cv::imshow("more connected pupil edges", moreConnectedPupilEdges);
	}
    //cv::imshow("raw input",input);


	//求两个glint的位置。
	//当在虹膜的范围内，只要出现了两个glint，就记住这两个glint的位置，作为当只有一个glint时的身份识别参考（一个glint离两个glint都存在的时候哪一个更近，就是哪一个）
	//如果反光斑超过2个，则选择离瞳孔更近的; 如果两个反光斑和瞳孔中心的连线构成的矢量夹角太小，则只取离瞳孔中心较近的那个。
	//每个红外灯照明可能在角膜上形成两个相聚很近的反光斑，永远取离瞳孔更近的。
	if(correctedExpandedPupilRoi.width>0 && correctedExpandedPupilRoi.height>0){
	    cv::Mat edgesInExpandedPupilRect;
		cv::Rect_<int> correctedExpandedPupilRoiInRawFrame;
		correctedExpandedPupilRoiInRawFrame.width = correctedExpandedPupilRoi.width*1.0 / scalingRatio;
		correctedExpandedPupilRoiInRawFrame.height = correctedExpandedPupilRoi.height*1.0 / scalingRatio;
		correctedExpandedPupilRoiInRawFrame.x = correctedExpandedPupilRoi.x*1.0 / scalingRatio;
		correctedExpandedPupilRoiInRawFrame.y = correctedExpandedPupilRoi.y*1.0 / scalingRatio;
	    cv::Mat correctedExpandedPupilImage = rawFrame(correctedExpandedPupilRoiInRawFrame);

		cv::Mat _dx,_dy,_blurred,_magnitude;
		ratio_threshold_canny(correctedExpandedPupilImage,0,0,0,0,_blurred,_dx,_dy,_magnitude,0.90,0.5,edgesInExpandedPupilRect);

		double minIntensity, maxIntensity;
		cv::minMaxLoc(_blurred, &minIntensity, &maxIntensity);
		//cv::imshow("expanded roied pupil in raw image",correctedExpandedPupilImage);
	    //cv::imshow("expanded roied pupil edges in raw image",edgesInExpandedPupilRect);

		//利用瞳孔反光斑边缘的亮度和对比度特征，去掉不合理的边缘。还没有用到形状约束。
		for(int i=0;i<edgesInExpandedPupilRect.cols;i++){
			for(int j=0;j<edgesInExpandedPupilRect.rows;j++){
				if(edgesInExpandedPupilRect.ptr<uchar>(j)[i]==255){
					cv::Point tempP(i,j);
					float meanContrast,meanLighterSideIntensity,meanDarkerSideIntensity;
					contrast_along_gradient(tempP,correctedExpandedPupilImage, _dx, _dy, 5,meanContrast,meanLighterSideIntensity,meanDarkerSideIntensity);
					//printf("point:(%d,%d); meanContrast: %f,meanLighterSideIntensity %f,meanDarkerSideIntensity %f\n",i,j,meanContrast,meanLighterSideIntensity,meanDarkerSideIntensity);
					//printf("point:(%d,%d); (maxIntensity, minIntensity):(%f,%f); meanContrast: %f,meanLighterSideIntensity %f\n",i,j,maxIntensity, minIntensity,(meanContrast-minIntensity)*255.0/(maxIntensity-minIntensity),(meanLighterSideIntensity-minIntensity)*255.0/(maxIntensity-minIntensity));
					if(!((meanLighterSideIntensity-minIntensity)*255.0/(maxIntensity-minIntensity)>200 && meanContrast>15)){ 
						edgesInExpandedPupilRect.ptr<uchar>(j)[i]=0;
					}
				}
			}
		}
		//cv::Mat kernel = cv::getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		//cv::morphologyEx(edgesInExpandedPupilRect, edgesInExpandedPupilRect, CV_MOP_CLOSE, kernel);
		//cv::imshow("glint candidates edge in raw image",edgesInExpandedPupilRect);

		//利用反光斑边缘的群体特征，重心处亮度大较大。内黑外亮的边缘。去掉短轴太短的边缘。
		std::vector<std::vector<cv::Point>> contours; 
		std::vector<std::vector<cv::Point>> selectedContours;
		cv::findContours(edgesInExpandedPupilRect,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		float pupilRadius = sqrt(pow(correctedPupilRoi.width/2,2)+pow(correctedPupilRoi.height/2,2))/1.414*1.0 / scalingRatio;

		for(int i=0;i<contours.size();i++){
			cv::RotatedRect glintRect = cv::minAreaRect(contours[i]);
			float minorAxisLength = glintRect.size.height>glintRect.size.width? glintRect.size.width:glintRect.size.height;
			float majorAxisLength = glintRect.size.height>glintRect.size.width? glintRect.size.height:glintRect.size.width;
			float pccrDistance;  //反光斑和瞳孔中心差向量的模长
			pccrDistance = sqrt(pow(glintRect.center.x + correctedExpandedPupilRoiInRawFrame.x - (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio,2) + 
			                    pow(glintRect.center.y + correctedExpandedPupilRoiInRawFrame.y - (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio,2));

			//printf("contours %d: minor axis length(%f); contour size: %lu\n",i,minorAxisLength,contours[i].size());
			if(minorAxisLength>2.9 &&  //glint 短轴长度至少有2个像素以上。
			   majorAxisLength > 4.9 &&
		       majorAxisLength < (correctedPupilRoi.height>correctedPupilRoi.width?correctedPupilRoi.height:correctedPupilRoi.width)*0.2*1.0 / scalingRatio &&  //长轴长度小于瞳孔直径的0.2
			   minorAxisLength/majorAxisLength>0.2  && //短长轴比大于0.3，不能太像直线 
			   pccrDistance < pupilRadius*1.8)  //瞳孔半径放大2倍的角膜范围内。
			{
			    int centerX=0;
			    int centerY=0;
			    for(int j=0;j<contours[i].size();j++){
			    	centerX +=contours[i][j].x;
			    	centerY +=contours[i][j].y;
					//printf("(%d,%d)\n",contours[i][j].x,contours[i][j].y);
			    }
			    centerX /= contours[i].size();
			    centerY /= contours[i].size();
				//printf("%dth center:(%d,%d) intensity: %d, balanced intensity:%f\n",i,centerX,centerY,correctedExpandedPupilImage.ptr<uchar>(centerY)[centerX],(correctedExpandedPupilImage.ptr<uchar>(centerY)[centerX]-minIntensity)*255.0/(maxIntensity-minIntensity));
			    if((correctedExpandedPupilImage.ptr<uchar>(centerY)[centerX]-minIntensity)*255.0/(maxIntensity-minIntensity)>220)    //glint 轮廓中心必须是亮斑。
			        selectedContours.push_back(contours[i]);
			}
		}

		cv::Mat glintEdgesInExpandedPupilRect(edgesInExpandedPupilRect.rows,edgesInExpandedPupilRect.cols,CV_8UC1,cv::Scalar(0));
		for(int i=0;i<selectedContours.size();i++){
			for(int j=0;j<selectedContours[i].size();j++){
				glintEdgesInExpandedPupilRect.ptr<uchar>(selectedContours[i][j].y)[selectedContours[i][j].x]=255;
			}
		}
		//printf("contours number: %lu\n",selectedContours.size());
		//cv::imshow("selected glint in raw image",glintEdgesInExpandedPupilRect);

		//利用反光斑的分布特性。
		//先要有双反光斑识别成功的初始化情况，双反光斑初始化时，要求瞳孔接近圆形。
		//然后识别单个反光斑时，要求单个反光斑在瞳孔和初始化双反光斑附近。
		//根据pupilCenterToGlintsCenterVector的左右关系去判断，如果有向量共线，则取摸长较小的
		std::vector<cv::Point2f> pupilCenterToGlintsCenterVector;
		for(int i=0;i<selectedContours.size();i++){
			float centerX=0;
			float centerY=0;
			cv::RotatedRect contoursPoistion = cv::minAreaRect(selectedContours[i]);
			centerX = contoursPoistion.center.x;
			centerY = contoursPoistion.center.y;
			pupilCenterToGlintsCenterVector.push_back(cv::Point2f(centerX + correctedExpandedPupilRoiInRawFrame.x - (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio,centerY+correctedExpandedPupilRoiInRawFrame.y-(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio)); //pccr
		}
		if(referenceInitalizedFrames>9000){  //双glint参考点过期。
			referenceInitializedForBothGlintsFlag=0;
			maxPupilSizeDuringTwoGlint=-10000;
		}

        std::vector<cv::Point2f> noncolinearGlintsToPupilCenter; //保存非共线的，y轴在瞳孔中心一定范围内的候选点。glint 减去pupilcenter的差向量。
		float ratio = ellipseFitted.size.width /ellipseFitted.size.height;
		//printf("ratio:%f\n",ratio);
		noncolinearGlintsToPupilCenter.push_back(cv::Point2f(-10000,-10000));
		noncolinearGlintsToPupilCenter.push_back(cv::Point2f(-10000,-10000));
		if(ratio>0.85 && ratio<(1.0/0.8)){ //满足收集双反光斑的条件：瞳孔接近圆形。 这种情况下，反光斑至少有一个。头戴式眼动仪只会随着头部肌肉上下滑动，反光斑的x值，不会随着滑动而剧烈变化，所以在原始图像中x轴的值可以用来判断此情况下单个反光斑是哪一边的。
			//全改，先找一个离瞳孔最近的，然后找和那个点水平线的。
			float mag2=10000;
			int closestGlintInd=-1;
			for(int i=0;i<pupilCenterToGlintsCenterVector.size();i++){
				float mag1 = sqrt(pow(pupilCenterToGlintsCenterVector[i].x,2)+pow(pupilCenterToGlintsCenterVector[i].y,2));
				if(mag1<mag2 && pupilCenterToGlintsCenterVector[i].y > -correctedPupilRoi.height*1.0 / scalingRatio/2.0*1.8 && pupilCenterToGlintsCenterVector[i].y < correctedPupilRoi.height*1.0 / scalingRatio/2.0*1.8 && mag1 < pupilRadius*1.8){
					noncolinearGlintsToPupilCenter[0].x = pupilCenterToGlintsCenterVector[i].x;
					noncolinearGlintsToPupilCenter[0].y = pupilCenterToGlintsCenterVector[i].y;
					closestGlintInd=i;
					mag2=mag1;
				}
			}

			mag2=10000;
			for(int i=0;i<pupilCenterToGlintsCenterVector.size();i++){
				if(i==closestGlintInd){
					continue;
				}
				float mag1 = sqrt(pow(pupilCenterToGlintsCenterVector[i].x,2)+pow(pupilCenterToGlintsCenterVector[i].y,2));
				if(mag1<mag2 && abs(pupilCenterToGlintsCenterVector[i].y -noncolinearGlintsToPupilCenter[0].y) < correctedPupilRoi.height*1.0 / scalingRatio*0.25  && abs(pupilCenterToGlintsCenterVector[i].x -noncolinearGlintsToPupilCenter[0].x) > correctedPupilRoi.width*1.0 / scalingRatio*0.5){
					noncolinearGlintsToPupilCenter[1].x = pupilCenterToGlintsCenterVector[i].x;
					noncolinearGlintsToPupilCenter[1].y = pupilCenterToGlintsCenterVector[i].y;
					mag2=mag1;
				}
			}
			if(noncolinearGlintsToPupilCenter[1].x<-9000)
			    noncolinearGlintsToPupilCenter.pop_back();			
			//调试信息
			for(int i=0;i<noncolinearGlintsToPupilCenter.size();i++){
				//printf("collect %lu glints %dth,(%f,%f)\n",noncolinearGlintsToPupilCenter.size(),i,noncolinearGlintsToPupilCenter[i].x,noncolinearGlintsToPupilCenter[i].y);
			}
			if(noncolinearGlintsToPupilCenter.size()>=2){
				referenceInitalizedFrames = 0;
				referenceInitializedForBothGlintsFlag=1;
				int tempSize = correctedPupilRoi.height>correctedPupilRoi.width?correctedPupilRoi.height:correctedPupilRoi.width;
				if(tempSize>maxPupilSizeDuringTwoGlint){
					maxPupilSizeDuringTwoGlint=tempSize;
				}

				if(noncolinearGlintsToPupilCenter[0].x < noncolinearGlintsToPupilCenter[1].x){
					leftGilntPositionReference.x=noncolinearGlintsToPupilCenter[0].x + (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;  //还原到原图坐标。
			        leftGilntPositionReference.y=noncolinearGlintsToPupilCenter[0].y + (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			        rightGilntPositionReference.x=noncolinearGlintsToPupilCenter[1].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        rightGilntPositionReference.y=noncolinearGlintsToPupilCenter[1].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;

					leftGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[0].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        leftGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[0].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[1].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[1].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
				}
				else{
					leftGilntPositionReference.x=noncolinearGlintsToPupilCenter[1].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        leftGilntPositionReference.y=noncolinearGlintsToPupilCenter[1].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			        rightGilntPositionReference.x=noncolinearGlintsToPupilCenter[0].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        rightGilntPositionReference.y=noncolinearGlintsToPupilCenter[0].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;

					leftGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[1].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        leftGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[1].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[0].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[0].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
				}
				cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(leftGilntPositionCurrent.x -correctedExpandedPupilRoiInRawFrame.x , leftGilntPositionCurrent.y-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(128),0,20,1,8);  //在correctedExpandedPupilRoi子图中显示
			    cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(rightGilntPositionCurrent.x -correctedExpandedPupilRoiInRawFrame.x , rightGilntPositionCurrent.y-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(255),0,20,1,8);
			}
			else{  //如果只有一个反光斑，就按照相对位置，判断是左边的，还是右边的。
			    if(noncolinearGlintsToPupilCenter.size()==1){
			        if(abs(noncolinearGlintsToPupilCenter[0].x+(correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio-leftGilntPositionReference.x)<abs(noncolinearGlintsToPupilCenter[0].x +(correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -rightGilntPositionReference.x)){ //判断单个反光斑，在原始图像中，离那个参考点更近
				    	leftGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[0].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			            leftGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[0].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			            rightGilntPositionCurrent.x=-10000;
			            rightGilntPositionCurrent.y=-10000;
				    	cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(noncolinearGlintsToPupilCenter[0].x + (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -correctedExpandedPupilRoiInRawFrame.x , noncolinearGlintsToPupilCenter[0].y+(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(128),0,20,1,8);
				    	//printf("for left\n");
				    }
				    else{
				    	leftGilntPositionCurrent.x=-10000;
			            leftGilntPositionCurrent.y=-10000;
			            rightGilntPositionCurrent.x=noncolinearGlintsToPupilCenter[0].x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			            rightGilntPositionCurrent.y=noncolinearGlintsToPupilCenter[0].y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
				    	cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(noncolinearGlintsToPupilCenter[0].x + (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -correctedExpandedPupilRoiInRawFrame.x , noncolinearGlintsToPupilCenter[0].y+(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(255),0,20,1,8);
				    	//printf("for right\n");
				    }
				}
			}
		}
		else{ //不满足双反光斑收集条件，即瞳孔不是接近圆形。即使有两个反光斑，也只收集一个反光斑。
		    cv::Point2f onlyOneGlint(-10000,-10000);
			float mag2=10000;
		    if(referenceInitializedForBothGlintsFlag && pupilCenterToGlintsCenterVector.size()>0){  //已经初始化识别成功了两个反光斑，就以这两个反光斑为基准。首先找到离瞳孔最近的反光斑，然后根据这个反光斑离哪个参考反光斑更近。
			    for(int i=0;i<pupilCenterToGlintsCenterVector.size();i++){
					float mag1 = sqrt(pow(pupilCenterToGlintsCenterVector[i].x,2)+pow(pupilCenterToGlintsCenterVector[i].y,2));
			    	if(mag1<mag2 &&
					   pupilCenterToGlintsCenterVector[i].y > -correctedPupilRoi.height*1.0 / scalingRatio/2.0*1.5 &&
					   pupilCenterToGlintsCenterVector[i].y <  correctedPupilRoi.height*1.0 / scalingRatio/2.0*1.5 && 
					   mag1<pupilRadius*1.5 &&
					   abs(pupilCenterToGlintsCenterVector[i].y+(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio-(leftGilntPositionReference.y+rightGilntPositionReference.y)/2.0) < correctedPupilRoi.height*1.5 / scalingRatio
					   ){  //瞳孔向下时，反光斑不超过瞳孔半径的1.5倍。瞳孔向上时，反光斑不超过瞳孔半径的1.5倍
					    onlyOneGlint.x = pupilCenterToGlintsCenterVector[i].x;
					    onlyOneGlint.y = pupilCenterToGlintsCenterVector[i].y;
					    mag2=mag1;
			    	}
			    }
		    }
			if(mag2>9000){    //无有效反光斑。
		        leftGilntPositionCurrent.x=-10000;
		    	leftGilntPositionCurrent.y=-10000;
		    	rightGilntPositionCurrent.x=-10000;
		    	rightGilntPositionCurrent.y=-10000;
			}
			else{
				if(abs(onlyOneGlint.x+(correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio-leftGilntPositionReference.x)<abs(onlyOneGlint.x +(correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -rightGilntPositionReference.x)){ //判断单个反光斑，在原始图像中，离那个参考点更近
					leftGilntPositionCurrent.x=onlyOneGlint.x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        leftGilntPositionCurrent.y=onlyOneGlint.y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.x=-10000;
			        rightGilntPositionCurrent.y=-10000;
					cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(onlyOneGlint.x + (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -correctedExpandedPupilRoiInRawFrame.x , onlyOneGlint.y+(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(128),0,20,1,8);
					//printf("only one glint for left\n");
				}
				else{
					leftGilntPositionCurrent.x=-10000;
			        leftGilntPositionCurrent.y=-10000;
			        rightGilntPositionCurrent.x=onlyOneGlint.x+ (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio;
			        rightGilntPositionCurrent.y=onlyOneGlint.y+ (correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio;
					cv::drawMarker(glintEdgesInExpandedPupilRect,cv::Point2i(onlyOneGlint.x + (correctedPupilRoi.x+correctedPupilRoi.width/2)*1.0 / scalingRatio -correctedExpandedPupilRoiInRawFrame.x , onlyOneGlint.y+(correctedPupilRoi.y+correctedPupilRoi.height/2)*1.0 / scalingRatio-correctedExpandedPupilRoiInRawFrame.y),cv::Scalar(255),0,20,1,8);
					//printf("only one glint for right\n");
				}
			}
		}
		//imshow("identified glints in raw roi image",glintEdgesInExpandedPupilRect);
	}
}

//用初始识别的边界点，来贪婪查找符合条件的点。
//递归法，如果某段边缘，selected points中某一部分断开了且没选中，然后碰上glint，边缘标准达不到，那么在detectedEdges中连续的后面的符合边缘标准的边缘也会被忽略掉。
//种子点期期初没有被加进来，但是种子点旁边的新点，添加邻居进来时，会把种子点加进来。
void PuRe::recursive_greedy_connectivity_search(std::vector<cv::Point>& output, cv::Point& p,cv::Mat detectedEdges, int contrastStep, float darkerSideIntensityHighLimit,float lighterSideIntensityHighLimit,float edgeContrastLowLimit){
	float tdx=dx.ptr<float>(p.y)[p.x];  //mat类型定了，最好用ptr访问。.data[]访问方式，不能指定数据类型，默认用char读取字节，读取其他数据类型时会出错。
	float tdy=dy.ptr<float>(p.y)[p.x];
	int connectNeighbourNum=3;

	for(int j=-connectNeighbourNum;j<=connectNeighbourNum;j++){
		for(int k=-connectNeighbourNum;k<=connectNeighbourNum;k++){
			if(j!=0 || k!=0){
			    int cx = p.x+j;
			    int cy = p.y+k;
			    float pupilEdgeContrast;
	            float pupilEdgeLighterSideIntensity;
	            float pupilEdgeDarkerSideIntensity; 
			    cv::Point tempP(cx,cy);
		        float angleCos;
		        if(cx>=0 && cx<detectedEdges.cols && cy>=0 && cy<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(cy)+(cx)]){
		        	if(sqrt(tdx*tdx+tdy*tdy)>0 && !(std::count(output.begin(),output.end(),tempP))){  //不包含已经添加到输出集中的点。
			    		angleCos = (tdx*j+tdy*k)/(sqrt(j*j+k*k)*sqrt(tdx*tdx+tdy*tdy));
		        		//printf("(px,py):(%d,%d); added point:(%d,%d); (tdx,tdy):(%f,%f);(j,k):(%d,%d); angleCos:%f\n",p.x,p.y,cx,cy,tdx,tdy,j,k,angleCos);
			    		if(angleCos<0.3){  //尽量沿着边缘的切向走，就是和法向尽量垂直。
			    		    if(contrast_along_gradient(tempP, input, dx, dy, contrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
								//printf("pupilEdgeContrast:%f (%f); pupilEdgeLighterSideIntensity:%f (%f); pupilEdgeDarkerSideIntensity:%f (%f) \n",pupilEdgeContrast,edgeContrastLowLimit,pupilEdgeLighterSideIntensity,lighterSideIntensityHighLimit,pupilEdgeDarkerSideIntensity,darkerSideIntensityHighLimit);
			    		    	if(pupilEdgeContrast>edgeContrastLowLimit && pupilEdgeDarkerSideIntensity<darkerSideIntensityHighLimit && pupilEdgeLighterSideIntensity< lighterSideIntensityHighLimit){
			    		    		output.push_back(tempP);
			    		    		recursive_greedy_connectivity_search(output, output.back(), detectedEdges, contrastStep,darkerSideIntensityHighLimit,lighterSideIntensityHighLimit, edgeContrastLowLimit);
			    		    	}
			    		    }
			    		}
		        	}
		        }
			}
		}
	}
}
//非递归法
//比递归法多考虑一步，就是种子点相连的原始边缘点串，全部会被纳入考虑范畴。
//这里输入进来的边缘都是单像素边缘，即每个边缘种子点最多有两个邻居。种子点中一定包含线段的端点，所以只对线段的端点开始循环。
void PuRe::direct_greedy_connectivity_search(std::vector<cv::Point>& output,
                                             std::vector<cv::Point>& pointsSeeds,
											 const cv::Mat& detectedEdges, 
											 int contrastStep, 
											 float darkerSideIntensityHighLimit,
											 float lighterSideIntensityHighLimit,
											 float edgeContrastLowLimit){
    output.clear();
	cv::Mat refinedEdgeFlag(detectedEdges.size(),CV_8UC1,cv::Scalar(0)); //0表示还没检验；1表示该点是refine后的边缘；255表示该点已经被检验，且不是边缘。
	float tdx;  //mat类型定了，最好用ptr访问。.data[]访问方式，不能指定数据类型，默认用char读取字节，读取其他数据类型时会出错。
	float tdy;
	int currentSeedPointX;
	int currentSeedPointY;
	float pupilEdgeContrast;
	float pupilEdgeLighterSideIntensity;
	float pupilEdgeDarkerSideIntensity;
	int neighboursNum;
	int edgesNum=0;  //detectedEdges进来时，可能pupil形成一个闭环，pointsSeeds都有两个邻居。

	for(int i=0;i<pointsSeeds.size();i++){
		currentSeedPointX = pointsSeeds[i].x;
		currentSeedPointY = pointsSeeds[i].y;
		neighboursNum=0;

		for(int ii=-1;ii<=1;ii++){        //找到种子的邻居，开始以邻居为起点，朝一个方向搜索所有的边缘点。最多只能有两个邻居。
			for(int jj=-1;jj<=1;jj++){
				if((ii!=0 || jj!=0) && detectedEdges.data[detectedEdges.cols*(currentSeedPointY+jj)+(currentSeedPointX+ii)])
				    neighboursNum++;
			}
		}
     
		if(refinedEdgeFlag.data[refinedEdgeFlag.cols*(currentSeedPointY)+(currentSeedPointX)] != 0  || neighboursNum!=1) //表示该点已经被检验过,或者不是端点
		    continue;
		edgesNum++;
		//printf("%dth seed point(x,y):(%d,%d); number of neighbours:%d; edge flag:%d\n",i,currentSeedPointX,currentSeedPointY,neighboursNum,refinedEdgeFlag.data[refinedEdgeFlag.cols*(currentSeedPointY)+(currentSeedPointX)]);

	    if(contrast_along_gradient(pointsSeeds[i], input, dx, dy, contrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
	    	if(pupilEdgeContrast>edgeContrastLowLimit && pupilEdgeDarkerSideIntensity<darkerSideIntensityHighLimit && pupilEdgeLighterSideIntensity< lighterSideIntensityHighLimit){
	    		output.push_back(pointsSeeds[i]);
				refinedEdgeFlag.data[refinedEdgeFlag.cols*(currentSeedPointY)+(currentSeedPointX)] = 1;
	    	}
			else
			    refinedEdgeFlag.data[refinedEdgeFlag.cols*(currentSeedPointY)+(currentSeedPointX)] = 255;
	    }
		else{
			refinedEdgeFlag.data[refinedEdgeFlag.cols*(currentSeedPointY)+(currentSeedPointX)] = 255;
		}
		
		std::vector<cv::Point> tempPointsList;
		int neighbourPointX = pointsSeeds[i].x;
		int neighbourPointY = pointsSeeds[i].y;
		bool continueSearchFlag;
		tempPointsList.push_back(pointsSeeds[i]);
		int indexInEdgeList=0;
		int edgePointsNum=0;

		do{
			continueSearchFlag=false;
	        for(int j=-1;j<=1;j++){
	        	for(int k=-1;k<=1;k++){
	        		if(j!=0 || k!=0){
	        		    int cx = neighbourPointX+j;
	        		    int cy = neighbourPointY+k;
	        		    cv::Point tempP(cx,cy);
	        	        if(cx>=0 && cx<detectedEdges.cols && cy>=0 && cy<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(cy)+(cx)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)]==0){
	        		    	if(contrast_along_gradient(tempP, input, dx, dy, contrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
	        		    		if(pupilEdgeContrast>edgeContrastLowLimit && pupilEdgeDarkerSideIntensity<darkerSideIntensityHighLimit && pupilEdgeLighterSideIntensity< lighterSideIntensityHighLimit){
	        		    			if(indexInEdgeList>3){
										output.push_back(tempP);
										edgePointsNum++;
									}
		    						refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)] = 1;
	        		    		}
	        		    	}							
							if(refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)] != 1)
						        refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)]=255;			

							neighbourPointX = cx;
							neighbourPointY = cy;
							tempPointsList.push_back(cv::Point(cx,cy));
							continueSearchFlag=true;
							indexInEdgeList++;
	        	        }				
	        		}
	        	}
	        }
		}while(continueSearchFlag);
		if(edgePointsNum>6){
			output.erase(output.end()-3,output.end()-1);
		}
		//printf("%dth edge list size:%d; start(x,y):(%d,%d); end(x,y):(%d,%d)\n",i,tempPointsList.size(),tempPointsList.begin()->x,tempPointsList.begin()->y,(tempPointsList.end()-1)->x,(tempPointsList.end()-1)->y);

        if(1){    //是否根据边缘末端曲率前进方向延伸,和末端邻域相连,解决边缘断开问题.
		    //tempPointsList中按顺序包含了一条边缘的所有的边界点。找到其中第一个和最后一个有效轮廓点。
		    int validPupilEdgeStartIndex=10000;
		    int validPupilEdgeEndIndex=-10000;
			int endExtendNewSeedCandidateSearchRadius=5;
			int endOffset=5;
			int maxCurvatureStep;
		    for(int ii=0;ii<tempPointsList.size();ii++){
		    	if(refinedEdgeFlag.data[refinedEdgeFlag.cols*(tempPointsList[ii].y)+(tempPointsList[ii].x)]==1){
		    		if(validPupilEdgeStartIndex>ii)
		    		    validPupilEdgeStartIndex=ii;
		    		if(validPupilEdgeEndIndex<ii)
		    		    validPupilEdgeEndIndex=ii;
		    	}
		    }
			int validEdgeLength = validPupilEdgeEndIndex-validPupilEdgeStartIndex+1;
		    if(validPupilEdgeEndIndex>-10000 && validPupilEdgeStartIndex<10000 && validEdgeLength>endOffset+2*maxCurvatureStep){  //太短了，算曲率容易不准
				//printf("valid edge start point(x,y):(%d,%d); end point(x,y):(%d,%d)\n",tempPointsList[validPupilEdgeStartIndex].x,tempPointsList[validPupilEdgeStartIndex].y,tempPointsList[validPupilEdgeEndIndex].x,tempPointsList[validPupilEdgeEndIndex].y);
		        std::vector<cv::Point> newEdgeCandidates;
				int curvatureExtendStep;

				//tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep]指向tempPointsList[validPupilEdgeStartIndex+endOffset]的向量为startCurvatureVector1
				//tempPointsList[validPupilEdgeStartIndex+endOffset+2*curvatureExtendStep]指向tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep]的向量为startCurvatureVector2
				//startCurvatureVector2•startCurvatureVector1构成曲率相关的偏转角的startCosValue
				//startCurvatureVector2×startCurvatureVector1构成曲率相关的偏转角的startSinValue
				//startCurvatureVector1经过startCosValue和startSinValue构成的矩阵去旋转，得到的点为向量终点，加上tempPointsList[validPupilEdgeStartIndex+endOffset]，即为曲率延伸后的候选点startEdgeExtendedPoints
				//最后在startEdgeExtendedPoints为中心的邻域内，找可能的瞳孔边缘点。

				/*建立基准坐标系，x向上，y向右，z垂直纸面向里，右手坐标系。
				    ^ x
					|
					|    __
					|      \
					|     \|/ 正角度方向   
					|   
					|-----------------> y

					由向量[x,y]变换到[x',y'], θ的依据正角度方向，由向量[x,y]指向[x',y']
					-   -   -                -  -  -
					| x'| = | cos(θ) -sin(θ) |  | x|  
					| y'|   | sin(θ)  cos(θ) |  | y|
					-   -   -                -  -  - 
				*/
			    
				if(1){    //在两个端点末端直接一次性按照曲率延伸最大20个像素。
                    //头部沿曲率方向延伸
					maxCurvatureStep=20;
					curvatureExtendStep = validEdgeLength/2 < maxCurvatureStep ? validEdgeLength/2 : maxCurvatureStep;

				    cv::Point startCurvatureVector1 = tempPointsList[validPupilEdgeStartIndex+endOffset] - tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep];
				    cv::Point startCurvatureVector2 = tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep] - tempPointsList[validPupilEdgeStartIndex+endOffset+2*curvatureExtendStep];
				    double startMagProduct =sqrt(startCurvatureVector1.x*startCurvatureVector1.x+startCurvatureVector1.y*startCurvatureVector1.y)*sqrt(startCurvatureVector2.x*startCurvatureVector2.x+startCurvatureVector2.y*startCurvatureVector2.y);
				    float startCosValue = (startCurvatureVector2.x*startCurvatureVector1.x + startCurvatureVector2.y*startCurvatureVector1.y)/startMagProduct;
				    float startSinValue = (startCurvatureVector2.x*startCurvatureVector1.y - startCurvatureVector2.y*startCurvatureVector1.x)/startMagProduct;
				    cv::Point startCurvatureExtendedPoint;
				    startCurvatureExtendedPoint.x =startCurvatureVector1.x*startCosValue - startCurvatureVector1.y*startSinValue +tempPointsList[validPupilEdgeStartIndex+endOffset].x;
				    startCurvatureExtendedPoint.y =startCurvatureVector1.x*startSinValue + startCurvatureVector1.y*startCosValue +tempPointsList[validPupilEdgeStartIndex+endOffset].y;
    
		            for(int ii=-endExtendNewSeedCandidateSearchRadius;ii<=endExtendNewSeedCandidateSearchRadius;ii++){
		            	for(int jj=-endExtendNewSeedCandidateSearchRadius;jj<=endExtendNewSeedCandidateSearchRadius;jj++){
		            		cv::Point temp;
		            		temp.x=startCurvatureExtendedPoint.x+ii;
		            		temp.y=startCurvatureExtendedPoint.y+jj;
		            		if(temp.x>=0 && temp.x<detectedEdges.cols && temp.y>=0 && temp.y<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(temp.y)+(temp.x)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(temp.y)+(temp.x)]==0){
				    			newEdgeCandidates.push_back(temp); 
		            		}
		            	}
		            }
                    //尾部沿曲率方向延伸
				    cv::Point endCurvatureVector1 = tempPointsList[validPupilEdgeEndIndex-endOffset] - tempPointsList[validPupilEdgeEndIndex-endOffset-curvatureExtendStep];
				    cv::Point endCurvatureVector2 = tempPointsList[validPupilEdgeEndIndex -endOffset- curvatureExtendStep] - tempPointsList[validPupilEdgeEndIndex -endOffset- 2*curvatureExtendStep];
				    double endMagProduct =sqrt(endCurvatureVector1.x*endCurvatureVector1.x+endCurvatureVector1.y*endCurvatureVector1.y)*sqrt(endCurvatureVector2.x*endCurvatureVector2.x+endCurvatureVector2.y*endCurvatureVector2.y);
				    float endCosValue = (endCurvatureVector2.x*endCurvatureVector1.x + endCurvatureVector2.y*endCurvatureVector1.y)/endMagProduct;
				    float endSinValue = (endCurvatureVector2.x*endCurvatureVector1.y - endCurvatureVector2.y*endCurvatureVector1.x)/endMagProduct;
				    cv::Point endCurvatureExtendedPoint;
				    endCurvatureExtendedPoint.x =endCurvatureVector1.x*endCosValue - endCurvatureVector1.y*endSinValue +tempPointsList[validPupilEdgeEndIndex-endOffset].x;
				    endCurvatureExtendedPoint.y =endCurvatureVector1.x*endSinValue + endCurvatureVector1.y*endCosValue +tempPointsList[validPupilEdgeEndIndex-endOffset].y;
    
		            for(int ii=-endExtendNewSeedCandidateSearchRadius;ii<=endExtendNewSeedCandidateSearchRadius;ii++){
		            	for(int jj=-endExtendNewSeedCandidateSearchRadius;jj<=endExtendNewSeedCandidateSearchRadius;jj++){
		            		cv::Point temp;
		            		temp.x=endCurvatureExtendedPoint.x+ii;
		            		temp.y=endCurvatureExtendedPoint.y+jj;
		            		if(temp.x>=0 && temp.x<detectedEdges.cols && temp.y>=0 && temp.y<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(temp.y)+(temp.x)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(temp.y)+(temp.x)]==0){
				    			newEdgeCandidates.push_back(temp);  
		            		}
		            	}
		            }
				}
				else{ //在两个端点末端直接多次按照曲率延伸，每次固定个像素。有bug！！！
				    maxCurvatureStep=5;
					curvatureExtendStep = validEdgeLength/2 < maxCurvatureStep ? validEdgeLength/2 : maxCurvatureStep;

				    //头部沿曲率方向延伸
				    cv::Point startCurvatureVector1 = tempPointsList[validPupilEdgeStartIndex+endOffset] - tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep];
				    cv::Point startCurvatureVector2 = tempPointsList[validPupilEdgeStartIndex+endOffset+curvatureExtendStep] - tempPointsList[validPupilEdgeStartIndex+endOffset+2*curvatureExtendStep];
				    double startMagProduct =sqrt(startCurvatureVector1.x*startCurvatureVector1.x+startCurvatureVector1.y*startCurvatureVector1.y)*sqrt(startCurvatureVector2.x*startCurvatureVector2.x+startCurvatureVector2.y*startCurvatureVector2.y);
				    float startCosValue = (startCurvatureVector2.x*startCurvatureVector1.x + startCurvatureVector2.y*startCurvatureVector1.y)/startMagProduct;
				    float startSinValue = (startCurvatureVector2.x*startCurvatureVector1.y - startCurvatureVector2.y*startCurvatureVector1.x)/startMagProduct;
				    
				    int curvatureExtendNums=4;
				    cv::Point startCurvatureExtendedReferencePoint = tempPointsList[validPupilEdgeStartIndex+endOffset];
				    cv::Point startCurvatureExtendedRotatedVector = startCurvatureVector1;
				    for(int extendIndex=0;extendIndex<curvatureExtendNums;extendIndex++){
				        startCurvatureExtendedRotatedVector.x =startCurvatureExtendedRotatedVector.x*startCosValue - startCurvatureExtendedRotatedVector.y*startSinValue;
				        startCurvatureExtendedRotatedVector.y =startCurvatureExtendedRotatedVector.x*startSinValue + startCurvatureExtendedRotatedVector.y*startCosValue;
				    	startCurvatureExtendedReferencePoint = startCurvatureExtendedReferencePoint + startCurvatureExtendedRotatedVector;
    
		                for(int ii=-endExtendNewSeedCandidateSearchRadius;ii<=endExtendNewSeedCandidateSearchRadius;ii++){
		                	for(int jj=-endExtendNewSeedCandidateSearchRadius;jj<=endExtendNewSeedCandidateSearchRadius;jj++){
		                		cv::Point temp;
		                		temp.x=startCurvatureExtendedReferencePoint.x+ii;
		                		temp.y=startCurvatureExtendedReferencePoint.y+jj;
		                		if(temp.x>=0 && temp.x<detectedEdges.cols && temp.y>=0 && temp.y<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(temp.y)+(temp.x)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(temp.y)+(temp.x)]==0){
				        			newEdgeCandidates.push_back(temp); 
		                		}
		                	}
		                }				
				    }

                    //尾部沿曲率方向延伸
				    cv::Point endCurvatureVector1 = tempPointsList[validPupilEdgeEndIndex- endOffset] - tempPointsList[validPupilEdgeEndIndex-endOffset-curvatureExtendStep];
				    cv::Point endCurvatureVector2 = tempPointsList[validPupilEdgeEndIndex -endOffset- curvatureExtendStep] - tempPointsList[validPupilEdgeEndIndex -endOffset- 2*curvatureExtendStep];
				    double endMagProduct =sqrt(endCurvatureVector1.x*endCurvatureVector1.x+endCurvatureVector1.y*endCurvatureVector1.y)*sqrt(endCurvatureVector2.x*endCurvatureVector2.x+endCurvatureVector2.y*endCurvatureVector2.y);
				    float endCosValue = (endCurvatureVector2.x*endCurvatureVector1.x + endCurvatureVector2.y*endCurvatureVector1.y)/endMagProduct;
				    float endSinValue = (endCurvatureVector2.x*endCurvatureVector1.y - endCurvatureVector2.y*endCurvatureVector1.x)/endMagProduct;
    
				    cv::Point endCurvatureExtendedReferencePoint = tempPointsList[validPupilEdgeEndIndex-endOffset];
				    cv::Point endCurvatureExtendedRotatedVector = endCurvatureVector1;
				    for(int extendIndex=0;extendIndex<curvatureExtendNums;extendIndex++){
				        endCurvatureExtendedRotatedVector.x =endCurvatureExtendedRotatedVector.x*endCosValue - endCurvatureExtendedRotatedVector.y*endSinValue;
				        endCurvatureExtendedRotatedVector.y =endCurvatureExtendedRotatedVector.x*endSinValue + endCurvatureExtendedRotatedVector.y*endCosValue;
				    	endCurvatureExtendedReferencePoint = endCurvatureExtendedReferencePoint + endCurvatureExtendedRotatedVector;
    
		                for(int ii=-endExtendNewSeedCandidateSearchRadius;ii<=endExtendNewSeedCandidateSearchRadius;ii++){
		                	for(int jj=-endExtendNewSeedCandidateSearchRadius;jj<=endExtendNewSeedCandidateSearchRadius;jj++){
		                		cv::Point temp;
		                		temp.x=endCurvatureExtendedReferencePoint.x+ii;
		                		temp.y=endCurvatureExtendedReferencePoint.y+jj;
		                		if(temp.x>=0 && temp.x<detectedEdges.cols && temp.y>=0 && temp.y<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(temp.y)+(temp.x)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(temp.y)+(temp.x)]==0){
				        			newEdgeCandidates.push_back(temp); 
		                		}
		                	}
		                }				
				    }
				}

		        //找到新拓展进来的边缘点的端点(起点或者终点)，将端点放入到pointsSeeds中，等待下一步的处理。
		        for(int ii=0;ii<newEdgeCandidates.size();ii++){
		        	cv::Mat newEdgeEndPointSearchFlag(detectedEdges.size(),CV_8UC1,cv::Scalar(0));
		        	int neighbourNum;
		        	int currentX = newEdgeCandidates[ii].x;
		        	int currentY = newEdgeCandidates[ii].y;
		        	newEdgeEndPointSearchFlag.data[newEdgeEndPointSearchFlag.cols*currentY+currentX]=1;
		        	int tempX;
		        	int tempY;

		        	do{

		        		neighbourNum = 0;
		        	    for(int jj=-1;jj<=1;jj++){
		        	    	for(int kk=-1;kk<=1;kk++){
		        	    		if((jj!=0 || kk!=0 ) && detectedEdges.data[newEdgeEndPointSearchFlag.cols*(currentY+kk)+currentX+jj] && newEdgeEndPointSearchFlag.data[newEdgeEndPointSearchFlag.cols*(currentY+kk)+currentX+jj] ==0 ){
		        	    			neighbourNum++;
		        					tempX = currentX+jj;
		        					tempY = currentY+kk;
		        	    		}
		        	    	}
		        	    }
		        		if(neighbourNum>=1){  //假设条件是单像素边缘,如果当前点是中间点,则朝一个方向延伸到起点就行.
		        			continueSearchFlag=true;
							newEdgeEndPointSearchFlag.data[newEdgeEndPointSearchFlag.cols*(currentY)+currentX]=1;
		        			currentX = tempX;
		        			currentY = tempY;
							//printf("end extend point(x,y):(%d,%d) from (x,y):(%d,%d)\n",currentX,currentY,newEdgeCandidates[ii].x,newEdgeCandidates[ii].y);
		        		}
		        		else{
		        			continueSearchFlag=false;
		        			if(!std::count(pointsSeeds.begin(),pointsSeeds.end(),cv::Point(currentX,currentY))){
		    					pointsSeeds.push_back(cv::Point(currentX,currentY));
		    					//printf("added seed point(x,y):(%d,%d)\n",currentX,currentY);
		    				}
							else{
								//printf("exist seed point(x,y):(%d,%d)\n",currentX,currentY);
							}
		        		}
		        	}while(continueSearchFlag);
		        }
		    }
	    }
	}

    //解决pupil轮廓没有断开，直接形成一个闭环的情况。
	if(edgesNum==0 && pointsSeeds.size()>0){   
		bool continueSearchFlag;
		int neighbourPointX = pointsSeeds[0].x;
		int neighbourPointY = pointsSeeds[0].y;

		if(contrast_along_gradient(pointsSeeds[0], input, dx, dy, contrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
	    	if(pupilEdgeContrast>edgeContrastLowLimit && pupilEdgeDarkerSideIntensity<darkerSideIntensityHighLimit && pupilEdgeLighterSideIntensity< lighterSideIntensityHighLimit){
	    		output.push_back(pointsSeeds[0]);
				refinedEdgeFlag.data[refinedEdgeFlag.cols*(neighbourPointY)+(neighbourPointX)] = 1;
	    	}
			else
			    refinedEdgeFlag.data[refinedEdgeFlag.cols*(neighbourPointY)+(neighbourPointX)] = 255;
	    }
		else{
			refinedEdgeFlag.data[refinedEdgeFlag.cols*(neighbourPointY)+(neighbourPointX)] = 255;
		}

		do{
			continueSearchFlag=false;
	        for(int j=-1;j<=1;j++){
	        	for(int k=-1;k<=1;k++){
	        		if(j!=0 || k!=0){
	        		    int cx = neighbourPointX+j;
	        		    int cy = neighbourPointY+k;
	        		    cv::Point tempP(cx,cy);
	        	        if(cx>=0 && cx<detectedEdges.cols && cy>=0 && cy<detectedEdges.rows && detectedEdges.data[detectedEdges.cols*(cy)+(cx)] && refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)]==0){
	        		    	if(contrast_along_gradient(tempP, input, dx, dy, contrastStep, pupilEdgeContrast, pupilEdgeLighterSideIntensity, pupilEdgeDarkerSideIntensity)){
	        		    		if(pupilEdgeContrast>edgeContrastLowLimit && pupilEdgeDarkerSideIntensity<darkerSideIntensityHighLimit && pupilEdgeLighterSideIntensity< lighterSideIntensityHighLimit){
	        		    			output.push_back(tempP);
		    						refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)] = 1;
	        		    		}
	        		    	}							
							if(refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)] != 1)
						        refinedEdgeFlag.data[refinedEdgeFlag.cols*(cy)+(cx)]=255;			

							neighbourPointX = cx;
							neighbourPointY = cy;
							continueSearchFlag=true;
	        	        }				
	        		}
	        	}
	        }
		}while(continueSearchFlag);
	}
}


//求某一个点沿着梯度方向，亮度高侧和亮度低侧的contrastStep个像素平均值的亮度差,暗侧亮度和亮侧亮度。
bool contrast_along_gradient(cv::Point& p,
                           cv::Mat& inputImage, 
						   cv::Mat& gradientX, 
						   cv::Mat& gradientY, 
						   int contrastStep,
						   float& meanContrast,
						   float& meanLighterSideIntensity,
						   float& meanDarkerSideIntensity){
	float dx = gradientX.ptr<float>(p.y)[p.x];
	float dy = gradientY.ptr<float>(p.y)[p.x];	
	float dMagnitude = sqrt(dx*dx+dy*dy);
    int sumedPointsNum=0;
	float deltaX;
	float deltaY;

	if(dMagnitude>0.001){  //防止分母为零出现。
	    deltaX = dx/dMagnitude;
	    deltaY = dy/dMagnitude;
	    meanContrast = 0;
	    meanLighterSideIntensity =0;
	    meanDarkerSideIntensity = 0;
		int xSign;
		int ySign;
		xSign = deltaX <0 ? -1:1; //deltaX为0是，算作符号为正
		ySign = deltaY <0 ? -1:1; //deltaY为0是，算作符号为正

	    if(abs(deltaX)>abs(deltaY)){  //直线插值，k>1,就按照每步y加1；k<1,就按照每步x加1
	    	float k = deltaY/deltaX;
	    	for(int i=1;i<=contrastStep;i++){
	    		if((p.x+i)<inputImage.cols && (p.x-i)>=0 && (p.y+i*abs(k))<inputImage.rows && (p.y-i*abs(k))>=0){
	    			meanLighterSideIntensity += inputImage.ptr<uchar>(p.y + (int)(i*abs(k)*ySign))[p.x + (int)(i*xSign)];
	    			meanDarkerSideIntensity += inputImage.ptr<uchar>(p.y - (int)(i*abs(k)*ySign))[p.x - (int)(i*xSign)];

	    			sumedPointsNum++;
	    		}
	    		else{
	    			break;
	    		}
	    	}
	    }
	    else{
	    	float _k = deltaX/deltaY;
	    	for(int i=1;i<=contrastStep;i++){
	    		if((p.y+i)<inputImage.rows && (p.y-i)>=0 && (p.x+i*abs(_k))<inputImage.cols && (p.x-i*abs(_k))>=0){
	    			meanLighterSideIntensity +=  inputImage.ptr<uchar>(p.y + (int)(i*ySign))[p.x + (int)(i*abs(_k)*xSign)];
	    			meanDarkerSideIntensity +=  inputImage.ptr<uchar>(p.y - (int)(i*ySign))[p.x - (int)(i*abs(_k)*xSign)];
	    			sumedPointsNum++;
	    		}
	    		else{
	    			break;
	    		}
	    	}
	    }
	    if(sumedPointsNum){	    			
	        meanLighterSideIntensity = meanLighterSideIntensity/sumedPointsNum;
	        meanDarkerSideIntensity = meanDarkerSideIntensity/sumedPointsNum;
	        meanContrast = meanLighterSideIntensity - meanDarkerSideIntensity;
	    }
	    else{
	    	meanLighterSideIntensity = -1;
	        meanDarkerSideIntensity = -1;
	        meanContrast = -1;
			return false;
	    }
	}
	else{
		meanLighterSideIntensity = -1;
	    meanDarkerSideIntensity = -1;
	    meanContrast = -1;
		return false;
	}
	return true;
}

// 得到图像的直方图
cv::MatND get_histogram(Mat &image)    
{  
    cv::MatND hist;  
    int channels[] = {0};  
    int dims = 1;  
    int histSize[] = {256};   
    float granges[] = {0, 255};  
    const float *ranges[] = {granges};  
    calcHist(&image, 1, channels, Mat(), hist, dims, histSize, ranges);  
    return hist;  
}  
 //  将图像直方图展示出来  
cv::Mat get_histogram_image(Mat &image) 
{  
    MatND hist = get_histogram(image);  
    Mat showImage(256,256, CV_8U,Scalar(0));  
    int i;  
    double maxValue = 0;  
    minMaxLoc(hist, 0, &maxValue, 0, 0);  
    for(i = 0; i < 256; i++)  
    {  
        float value = hist.at<float>(i);  
        int intensity = saturate_cast<int>(256 - 256* (value/maxValue));  
        rectangle(showImage, Point(i,256 - 1), Point((i+1)-1, intensity), Scalar(255));  
    }  
    return showImage;  
}

//用不同的颜色显示线条。
void show_lines_random_color(cv::Mat& bg, bool drawLinesInBg, std::vector<std::vector<cv::Point2i>>& lines, cv::String windowName){
	cv::namedWindow(windowName);
	cv::RNG rng(12345);
	cv::Mat showFilteredEdge(bg.rows,bg.cols,CV_8UC3,cv::Scalar(0,0,0));
	if(drawLinesInBg){
		bg.copyTo(showFilteredEdge);
	}
	for(int i=0;i<lines.size();i++){
		cv::Scalar color;
		while(1){
			color=cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			if((color[0]+color[1]+color[2])/3.0 >127 && (color[0]+color[1]+color[2])/3.0 <240)
			    break;
		}
		for(int j=0;j<lines[i].size();j++){
			showFilteredEdge.at<cv::Vec3b>(lines[i][j].y,lines[i][j].x) = cv::Vec3b(color[0],color[1],color[2]);
		}			
	}
	cv::imshow(windowName,showFilteredEdge);
}


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
                           ){
	if (!useInputGradient) {  //默认blurImage为true
		Size blurSize(_blurSize,_blurSize);
		inputGradient_x = Mat::zeros(grayImageInput.size(), CV_32F);
	    inputGradient_y = Mat::zeros(grayImageInput.size(), CV_32F);
	    inputMagnitude = Mat::zeros(grayImageInput.size(), CV_32F);

		if(blurFlag){
            GaussianBlur(grayImageInput, blurred, blurSize, _blurSigam, _blurSigam, BORDER_REPLICATE);   //高斯模糊	
		}
		else{
			blurred=grayImageInput;
		}
		
		Sobel(blurred, inputGradient_x, inputGradient_x.type(), 1, 0, 7, 1, BORDER_REPLICATE);   //求x方向偏导
	    Sobel(blurred, inputGradient_y, inputGradient_y.type(), 0, 1, 7, 1, BORDER_REPLICATE);   //求y方向偏导
		cv::magnitude(inputGradient_x, inputGradient_y, inputMagnitude);     //求图像2D梯度模长
	}
	cv::Mat edgeType = Mat::zeros(grayImageInput.size(), CV_8U);
	outputEdge = Mat::zeros(grayImageInput.size(), CV_8U);
	/*
	 *  Magnitude
	 * 1、求梯度直方图
	 * 2、8方向梯度非极大抑制
	 */
	double minMag = 0;    //图像梯度最大值
	double maxMag = 0;    //图像梯度最小值
	float *p_res;                        //p_res：magnitude矩阵的某一行的首地址。
	float *p_x, *p_y; // result, x, y          p_x: x偏导矩阵某一行的向量首地址；p_y：y偏导矩阵某一行的向量首地址；
	cv::minMaxLoc(inputMagnitude, &minMag, &maxMag);  //找到图像梯度模长最大和最小的值

	/*
	 *  Threshold selection based on the magnitude histogram
	 */
	float low_th = 0;   //识别为边缘的下限梯度值，不是从直方图来，而是高阈值的40%
	float high_th = 0;  //梯度的强度是否识别为边缘的上限阈值

	// Normalization
	inputMagnitude = inputMagnitude / maxMag;  //归一化梯度。 https://www.zhihu.com/question/20467170

	// Histogram
	int bins=256;
	int *histogram = new int[bins]();     //梯度直方图
	Mat res_idx = (bins-1) * inputMagnitude;  //将最大的梯度模长安排到数组索引最大的bin上。
	res_idx.convertTo(res_idx, CV_16U);
	short *p_res_idx=0;
	for(int i=0; i<res_idx.rows; i++){
		p_res_idx = res_idx.ptr<short>(i);
		for(int j=0; j<res_idx.cols; j++)
			histogram[ p_res_idx[j] ]++;
	}

	// Ratio
	int sum=0;
	int nonEdgePixels = notEdgePixRatio * grayImageInput.rows * grayImageInput.cols;  //默认有70% 的梯度强度不够成为边缘，这是自适应边缘的一种方法。只有30%的像素位置的梯度能够成为边缘。
	for(int i=0; i<bins; i++){
		sum += histogram[i];
		if( sum > nonEdgePixels ){
			high_th = float(i+1) / bins ;
			break;
		}
	}
	low_th = lowHighThresholdRatio*high_th;  // 这个地方很诡异，low_th也是为边缘识别设立的么？降低了梯度成为边缘的要求。

	delete[] histogram;

	/*
	 *  Non maximum supression
	 * 将梯度在上半平面，分为45度角附近区域，135度角附近区域，0度角附近区域，90度角附近区域，4个区域，中间点的梯度要比沿着该区域直线的前后两点的梯度要大，就记录为边缘
	 * 大于高阈值记为255强边缘，大于低阈值的记为128若边缘。
	 */
	const float tg22_5 = 0.4142135623730950488016887242097f;
	const float tg67_5 = 2.4142135623730950488016887242097f;
	uchar *_edgeType;           //指向edgeType的矩阵的某一行的首地址。
	float *p_res_b, *p_res_t;   //梯度矩阵的某一行的上下两行的行向量的首地址。
	edgeType.setTo(0);
	for(int i=1; i<inputMagnitude.rows-1; i++) {

		_edgeType = edgeType.ptr<uchar>(i);

		p_res=inputMagnitude.ptr<float>(i);     //梯度矩阵某一行的首地址
		p_res_t=inputMagnitude.ptr<float>(i-1);  //梯度矩阵前一行的首地址
		p_res_b=inputMagnitude.ptr<float>(i+1);  //梯度矩阵后一行的首地址

		p_x=inputGradient_x.ptr<float>(i);
		p_y=inputGradient_y.ptr<float>(i);

		for(int j=1; j<inputMagnitude.cols-1; j++){

			float m = p_res[j];
			if (m < low_th)          //当前点的梯度强度低于低限阈值，则不处理该点。
				continue;

			float iy = p_y[j];
			float ix = p_x[j];
			float y  = abs( (double) iy );
			float x  = abs( (double) ix );

			uchar val = p_res[j] > high_th ? 255 : 128;  //梯度强度大于高阈值，则为边界类型为255； 大于低阈值，小于高阈值，则边界类型为128。

			float tg22_5x = tg22_5 * x;
			if (y < tg22_5x) {                    //梯度方向在 0~22.5度范围内
				if (m > p_res[j-1] && m >= p_res[j+1])  //如果梯度值比左右的都大，记录该点的边界类型。
					_edgeType[j] = val;
			} else {
				float tg67_5x = tg67_5 * x;
				if (y > tg67_5x) {              //梯度方向在 67.5 ~90度
					if (m > p_res_b[j] && m >= p_res_t[j])  //如果梯度比上下的都大，则记录该点的边界类型。
						_edgeType[j] = val;
				} else {                        //梯度方向为 22.5~67.5度之间，
					if ( (iy<=0) == (ix<=0) ) {           //斜率为正
						if ( m > p_res_t[j-1] && m >= p_res_b[j+1])  //比上一行的前一个大，比下一行的后一个大，梯度的主方向为45度角附近。
							_edgeType[j] = val;
					} else {                             //斜率为负  
						if ( m > p_res_b[j-1] && m >= p_res_t[j+1])   //比下一行的前一个大，比上一行的后一个大，梯度的主方向为-45度角附近。
							_edgeType[j] = val;
					}
				}
			}
		}
	}

	/*
	 *  Hystheresis
	 * 
	 * 将边缘为255的种子点和与它相连的点（边缘为128或者255），保存到了edge中。
	 * 
	 * 将孤立的或者成批的128的边缘忽略。
	 */
	int pic_x=edgeType.cols;  //边缘类型宽
	int pic_y=edgeType.rows;  //边缘类型高
	int area = pic_x*pic_y;   //边缘图面积
	int lines_idx=0;
	int idx=0;  //不断加上pic_x，用来二维图像一维化访问过程中的换行。

	vector<int> lines;  //用来临时保存线段。
	outputEdge.setTo(0);  //最终保存数据结果。
	for(int i=1;i<pic_y-1;i++){               //外围留了一圈，一个像素。
		for(int j=1;j<pic_x-1;j++){           //外围留了一圈，一个像素。

			if( edgeType.data[idx+j] != 255 || outputEdge.data[idx+j] != 0 )   // edgeType.data[idx+j] == 255 &&  edge.data[idx+j] == 0, 就不会continue，满足条件，执行循环体。 忽略了弱边缘128的种子点。
				continue;

			outputEdge.data[idx+j] = 255;   //edgeType.data[idx+j] == 255 &&  edge.data[idx+j] == 0才会执行这一步。 相当于是强边缘，直接记录为边。idx中已经包含了行的整数倍。  
			lines_idx = 1;
			lines.clear();
			lines.push_back(idx+j);  //当前线段上记录这一个点
			int akt_idx = 0;

			while(akt_idx<lines_idx){   //相当于是在做一个米字形生长，以一个种子点开始，只要旁边8个点满足条件，就加入到直线中来。
				int akt_pos=lines[akt_idx];
				akt_idx++;

				if( akt_pos-pic_x-1 < 0 || akt_pos+pic_x+1 >= area )  //akt_pos超出图像处于，就提前结束循环。左后方的点线性坐标小于0,或者右前方的点，线性坐标大于总像素。
					continue;

				for(int k1=-1;k1<2;k1++)         //对某一个点进行米字型连接分析，如果边界数据不  edge.data[(akt_pos+(k1*pic_x))+k2] ==0 && edgeType.data[(akt_pos+(k1*pic_x))+k2] != 0 就进行后续分析
					for(int k2=-1;k2<2;k2++){
						if(outputEdge.data[(akt_pos+(k1*pic_x))+k2]!=0 || edgeType.data[(akt_pos+(k1*pic_x))+k2]==0)
							continue;
						outputEdge.data[(akt_pos+(k1*pic_x))+k2] = 255;   //将此点的边界置为255，
						lines.push_back((akt_pos+(k1*pic_x))+k2);   //将此点加入到线的vector中。
						lines_idx++;
					}
			}
		}
		idx+=pic_x;  //换到下一行
	}
}

/*在图像是单像素边缘时，findContour找到的线条，都是环，如果是单条线，也是一来一去,算了两次,起点不一定在两端。
如果找到的contour是环，那么会有重复的轮廓（一个轮廓完全覆盖另一个轮廓），被以不同的点存储顺序，出现了两遍。BUG!!!
本函数的功能是：
1、假设全图的边缘都是单像素边缘，每个点最多有2个邻居。先找非封闭线段，起点为端点。
2、再找封闭线段，起点为第一扫描到的点。
*/
void get_line_points_lists_from_edge(cv::Mat& edges, std::vector<std::vector<cv::Point>>& linePointsLists){
	
	cv::Mat edgeFlag(edges.rows, edges.cols, CV_8U,cv::Scalar(0));

	for(int i=1;i<edges.cols-1;i++){   //x方向水平，用i递增；y方向竖直，用j递增。  找到线段
		for(int j=1;j<edges.rows-1;j++){
			std::vector<cv::Point> linePoints;
			if(edges.data[j*edges.cols+i] && !(edgeFlag.data[j*edges.cols+i])){  //遍历，到某个点，如果是还没利用的边界，就开始分析。
				int neighbourNum;
				int currentPointX=i;
				int currentPointY=j;
				linePoints.clear();
				do{
					neighbourNum=0;
					int deltaX,deltaY;
			        for(int m=-1;m<=1;m++){
			        	for(int n=-1;n<=1;n++){
							if(m!=0 || n!=0){
				    		    if(edges.data[(currentPointY+m)*edges.cols+(currentPointX+n)] && !(edgeFlag.data[(currentPointY+m)*edges.cols+(currentPointX+n)])){
				    		    	neighbourNum++;
							    	deltaX=n;
							    	deltaY=m;
				    		    }								
							}
			        	}
			        }
					if(neighbourNum<=1){ //起点或者线段中间的点或者终点。
					    linePoints.push_back(cv::Point2i(currentPointX,currentPointY));
						edgeFlag.data[(currentPointY)*edges.cols+(currentPointX)]=1;

					}
					else{
						//printf("too many neighbours: %d neighbours at point (%d,%d)\n",neighbourNum,currentPointX,currentPointY);
					}
					currentPointX += deltaX;
					currentPointY += deltaY;
				}while(neighbourNum==1);
				if(linePoints.size()>1){  //过滤掉孤立的点。
					linePointsLists.push_back(linePoints);
				}
			}
		}
	}

	for(int i=1;i<edges.cols-1;i++){   //x方向水平，用i递增；y方向竖直，用j递增。  找到封闭图形。现在只剩下封闭图形。
		for(int j=1;j<edges.rows-1;j++){
			std::vector<cv::Point> linePoints;
			if(edges.data[j*edges.cols+i] && !(edgeFlag.data[j*edges.cols+i])){  //遍历，到某个点，如果是还没利用的边界，就开始分析。
				int neighbourNum;
				int currentPointX=i;
				int currentPointY=j;
				linePoints.clear();
				do{
					neighbourNum=0;
					int deltaX,deltaY;
			        for(int m=-1;m<=1;m++){
			        	for(int n=-1;n<=1;n++){
							if(m!=0 || n!=0){
				    		    if(edges.data[(currentPointY+m)*edges.cols+(currentPointX+n)] && !(edgeFlag.data[(currentPointY+m)*edges.cols+(currentPointX+n)])){
				    		    	neighbourNum++;
							    	deltaX=n;
							    	deltaY=m;
									break;   //封闭图形的所有点都是只有两个邻居。沿着任意其中一个方向开始延伸。起点是随机的，然后在kcos分割中去找到最佳的起点。
				    		    }								
							}
			        	}
			        }
					if(neighbourNum<=1){ 
					    linePoints.push_back(cv::Point2i(currentPointX,currentPointY));
						edgeFlag.data[(currentPointY)*edges.cols+(currentPointX)]=1;

					}
					else{
						//printf("too many neighbours: %d neighbours at point (%d,%d)\n",neighbourNum,currentPointX,currentPointY);
					}
					currentPointX += deltaX;
					currentPointY += deltaY;
				}while(neighbourNum==1);
				if(linePoints.size()>1){  //过滤掉孤立的点。
					linePointsLists.push_back(linePoints);
				}
			}
		}
	}
}


void point_mirror_about_axis(cv::Point& sourcePoint, cv::Point& mirroredPoint, cv::Point& axisPoint, float axisX, float axisY){
	//householder变换，H=I-2*w*w^T，w是某一个镜像面或者的轴的垂直向量。
	//二维的时候，也可使用镜像轴i和原向量x的内积关系。 x' = x•i*i - (x-x•i*i) = 2*x•i*i - x;
	//平面上的点,在进行向量镜像前,要构建成以镜像轴上某点为坐标原点(向量起点)为起点的向量.
	double mag = sqrt(axisX*axisX+axisY*axisY);
	axisX /= mag;
	axisY /= mag;
	double dot = (sourcePoint.x-axisPoint.x)*axisX + (sourcePoint.y-axisPoint.y)*axisY;
	mirroredPoint.x = 2*dot*axisX - sourcePoint.x + 2*axisPoint.x;
	mirroredPoint.y = 2*dot*axisY - sourcePoint.y + 2*axisPoint.y;
}

/*
瞳孔和反光斑识别总结：
1、opencv自带的canny和PuRe修改的canny，主要区别在两个梯度阈值的设置。前者是绝对值，后者是假设全图的梯度中，边缘占比数，通过直方图来寻找动态的阈值。
2、得到边缘后，进行后处理很重要。太多的边缘会有太多的邻居边缘点，导致异常的链接，在findContour时，得不到想要的轮廓。可以先边缘thinner处理，或者先findContour，对contours结果断开分析。
3、断开后的边缘，在通过边缘的对比度，亮侧平均亮度，暗侧平均亮度，进行聚合。例如利用本项目中的greedyConnectivitySearch算法。
4、根据先验假设的边缘所属对象的特性，例如大小，外截矩形长短轴比例，拟合后的椭圆长短轴比例，所处的大致位置，进行筛选。
5、findContour中的参数 hierarchyGlint,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE，保留轮廓的拓扑信息很重要。
6、本项目中，PuRe寻找瞳孔时，得到瞳孔的候选的算法有待改进（findPupilEdgeCandidates  combineEdgeCandidates）。例如瞳孔断开严重时，只能识别部分瞳孔，或者将虹膜的圆弧边缘纳入进来了。
7、图像识别相关的问题，永远优先用的是边缘和梯度，不到万不得已，不得用灰度值。
8、瞳孔的算法中，候选的瞳孔边缘，错误纳入进来虹膜边缘，需要进步一处理。
9、沿着曲率延伸边缘，用来解决瞳孔边缘断开过多，或者睫毛的影响。睫毛太短，瞳孔太大时，虹膜边缘会被引入,最好提前做闭环检测，闭环当做瞳孔。
10、输出的currentPupilConfidence目前用的还是pure输处的，要根据refine后的结果，进一步确定currentPupilConfidence。
*/
