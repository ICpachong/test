/*
 * Project:  - kptsComparison - Comparison of keypoint detection and matching, 
 *           including SIFT, ORB, and FAST from OpenCV
 *           and FFME from Carlos Roberto del Blanco
			本程序对SIFT ,ORB, FAST, FFME四种特种匹配的算法进行比较，从使用的结果可以看到FFME的速度非常的快，但是效果一般
 *
 * File:     main.cpp
 *
 * Contents: Comparison of kpoints detection and matching methods
 *
 * Authors:   Marcos Nieto <mnieto@vicomtech.org>
 *                - OpenCV 2.x API wrapper
 *                - main.cpp
 *            Carlos Roberto del Blanco <cda@gti.ssr.upm.es>
 *                - FFME algorithm
 *
 */

#ifdef WIN32
	#include <windows.h>
	#include "conio.h"
	#include <time.h>
#endif

#ifdef linux
	#include <stdio.h>
	#include <sys/time.h>
	#include <time.h>
#endif

#include <iostream>
#include <stdexcept>

#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "FFME.h"

using namespace std;
using namespace cv;

// Timing
#ifdef WIN32
	double t1, t2;	
#else
	int t1, t2;	
	struct timeval ts;
#endif
double t;

// Storage of 
FileStorage fs;

namespace
{
    void drawMatchesRelative(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        std::vector<cv::DMatch>& matches, Mat& img, const vector<unsigned char>& mask = vector<
        unsigned char> ())
    {
        for (int i = 0; i < (int)matches.size(); i++)
        {
            if (mask.empty() || mask[i])
            {
                Point2f pt_new = query[matches[i].queryIdx].pt;
                Point2f pt_old = train[matches[i].trainIdx].pt;

                cv::line(img, pt_new, pt_old, Scalar(0, 0, 255), 2);
                cv::circle(img, pt_new, 2, Scalar(255, 0, 0), 1);

            }
        }
    }

    //Takes a descriptor and turns it into an xy point
    void keypoints2points(const vector<KeyPoint>& in, vector<Point2f>& out)
    {
        out.clear();
        out.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i)
        {
            out.push_back(in[i].pt);
        }
    }

    //Takes an xy point and appends that to a keypoint structure
    void points2keypoints(const vector<Point2f>& in, vector<KeyPoint>& out)
    {
        out.clear();
        out.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i)
        {
            out.push_back(KeyPoint(in[i], 1));
        }
    }

    //Uses computed homography H to warp original input points to new planar position
    void warpKeypoints(const Mat& H, const vector<KeyPoint>& in, vector<KeyPoint>& out)
    {
        vector<Point2f> pts;
        keypoints2points(in, pts);
        vector<Point2f> pts_w(pts.size());
        Mat m_pts_w(pts_w);
        perspectiveTransform(Mat(pts), m_pts_w, H);
        points2keypoints(pts_w, out);
    }

    //Converts matching indices to xy points
    void matches2points(const vector<KeyPoint>& train, const vector<KeyPoint>& query,
        const std::vector<cv::DMatch>& matches, std::vector<cv::Point2f>& pts_train,
        std::vector<Point2f>& pts_query)
    {

        pts_train.clear();
        pts_query.clear();
        pts_train.reserve(matches.size());
        pts_query.reserve(matches.size());

        size_t i = 0;

        for (; i < matches.size(); i++)
        {

            const DMatch & dmatch = matches[i];

            pts_query.push_back(query[dmatch.queryIdx].pt);
            pts_train.push_back(train[dmatch.trainIdx].pt);

        }

    }

    void resetH(Mat&H)
    {
        H = Mat::eye(3, 3, CV_32FC1);
    }
}

// Class Arguments that contain all the potential arguments of the program
class Arguments
{
public:
	Arguments();
	static Arguments read(int argc, char** argv);
		
	// Video
	char *videoFileName;		
	bool saveVideoOut;
	bool playMode;
	bool verbose;	
	int procWidth;

	// Max. Num. Kpts.
	int maxNumKpts;
	
};

// Class Application that gathers the functionalities of the program
class Application
{
public:
	Application(const Arguments &s);
	void run();

	void handleKey(char key);

	void ffmeWorkBegin();
	void ffmeWorkEnd();
	string ffmeWorkFps() const;

	void siftWorkBegin();
	void siftWorkEnd();
	string siftWorkFps() const;

	void orbWorkBegin();
	void orbWorkEnd();
	string orbWorkFps() const;

	void fastWorkBegin();
	void fastWorkEnd();
	string fastWorkFps() const;

	void workBegin();
	void workEnd();
	string workFps() const;

	string message() const;

	// Main processing function
	int work(cv::Mat &inputImg, int frameNum);

private:
	Application operator=(Application&);

	Arguments arguments;
	
	// Video
	cv::Size procSize;
	bool running;
	std::string videoFileNameOut;
	int fourccOut;
	bool useCamera;	

	// SIFT
	cv::Mat SIFT_outputImg;
	cv::Mat SIFT_H_prev;
	cv::Ptr<cv::FeatureDetector> SIFT_detector;
	std::vector<cv::KeyPoint> SIFT_train_kpts, SIFT_query_kpts;
	std::vector<Point2f> SIFT_train_pts, SIFT_query_pts;
	
	cv::Ptr<cv::DescriptorExtractor> SIFT_descriptor;
	cv::Mat SIFT_train_desc, SIFT_query_desc;
	
	cv::Ptr<cv::DescriptorMatcher> SIFT_matcher;
	std::vector<cv::DMatch> SIFT_matches;
	std::vector<unsigned char> SIFT_match_mask;	

	// ORB
	cv::Mat ORB_outputImg;
	cv::Mat ORB_H_prev;
	cv::Ptr<cv::OrbFeatureDetector> ORB_detector;
	//cv::Ptr<cv::FeatureDetector> ORB_detector;
	std::vector<cv::KeyPoint> ORB_train_kpts, ORB_query_kpts;
	std::vector<Point2f> ORB_train_pts, ORB_query_pts;

	cv::Ptr<cv::DescriptorExtractor> ORB_descriptor;
	cv::Mat ORB_train_desc, ORB_query_desc;

	cv::Ptr<cv::DescriptorMatcher> ORB_matcher;
	std::vector<cv::DMatch> ORB_matches;
	std::vector<unsigned char> ORB_match_mask;	

	// FAST
	cv::Mat FAST_outputImg;
	cv::Mat FAST_H_prev;
	cv::Ptr<cv::FeatureDetector> FAST_detector;
	std::vector<cv::KeyPoint> FAST_train_kpts, FAST_query_kpts;
	std::vector<Point2f> FAST_train_pts, FAST_query_pts;

	cv::Ptr<cv::DescriptorExtractor> FAST_descriptor;
	cv::Mat FAST_train_desc, FAST_query_desc;

	cv::Ptr<cv::DescriptorMatcher> FAST_matcher;
	std::vector<cv::DMatch> FAST_matches;
	std::vector<unsigned char> FAST_match_mask;

	// FFME
	cv::Mat FFME_outputImg;
	FFME ffme;
	std::vector<cv::KeyPoint> FFME_train_kpts, FFME_query_kpts;
	std::vector<Point2f> FFME_train_pts, FFME_query_pts;
	cv::Mat FFME_train_desc, FFME_query_desc;
	std::vector<cv::DMatch> FFME_matches;
	
			
	// Time control
	int64 sift_work_begin;
	double sift_work_fps;
	int64 sift_work_ms;

	int64 orb_work_begin;
	double orb_work_fps;
	int64 orb_work_ms;

	int64 fast_work_begin;
	double fast_work_fps;
	int64 fast_work_ms;

	int64 ffme_work_begin;
	double ffme_work_fps;
	int64 ffme_work_ms;

	int64 work_begin;
	double work_fps;
};

static void help()
{
	 cout << "/*\n"
         	 << " **************************************************************************************************\n"
	 	 << " * Comparison of SIFT, FAST, ORB and FFME \n"
         	 << " * ----------------------------------------------------\n"		 
		 << " * \n"
		 << " * Author:Marcos Nieto and Carlos Roberto del Blanco\n"
		 << " * www.vicomtech.org, www.gti.ssr.upm.es\n"
		 << " * mnieto@vicomtech.org - cda@gti.ssr.upm.es\n"
		 << " * \n"
		 << " * \n"
		 << " * Date:13/07/2012\n"
		 << " **************************************************************************************************\n"
		 << " * \n"
		 << " * Usage: \n"		 		 
		 << " *		-video		# Specifies video file as input (if not specified, camera is used) \n"		 
		 << " *		-videoOut       # Specifies ON, OFF the recording of video\n"
		 << " *		-verbose	# Actives verbose: ON, OFF (default)\n"		 
		 << " *		-resizedWidth	# Specifies the desired width of the image (the height is computed to keep aspect ratio)\n"
		 << " *		-maxNumKpts	# Specifies the maximum number of points (default is 500)\n"
		 << " *\n"
		 << " * Example:\n"
		 << " *		kptsComparison -video myVideo.avi -verbose ON\n"
		 << " *		kptsComparison -resizedWidth 300\n"
		 << " * \n"
		 << " * Keys:\n"
		 << " *		Esc: Quit\n"
         << " */\n" << endl;
}

/** Main function*/
int main(int argc, char** argv)
{
	try
	{		
		help();		// Show help		
		if(argc < 1)
			return -1;
		Arguments arguments = Arguments::read(argc, argv); // Parse arguments
		Application application(arguments); // Init app
		application.run();	// Run app
	}
	catch (const exception& e) { return cout << "error: " << e.what() << endl, 1; }
	catch (...) { return cout << "unknown exception" << endl, 1; }
	return 0;
}

// Default constructor
Arguments::Arguments()
{
	// Default arguments
	videoFileName = 0;	
	procWidth = 0;	
	saveVideoOut = false;
	playMode = false;
	verbose = false;

	maxNumKpts = 1000;
	
}
// Argument parsing
Arguments Arguments::read(int argc, char **argv)
{
	Arguments arguments;

	for(int i=1; i<argc; i++)
	{
		if     (string(argv[i]) == "-video")       arguments.videoFileName = argv[++i]; 
		else if(string(argv[i]) == "-procWidth")  arguments.procWidth = atoi(argv[++i]);
		else if(string(argv[i]) == "-verbose")       arguments.verbose = (string(argv[++i]) == "ON");
		else if(string(argv[i]) == "-videoOut")      arguments.saveVideoOut = (string(argv[++i]) == "ON");				
		else if(string(argv[i]) == "-maxNumKpts")	arguments.maxNumKpts = atoi(argv[++i]);
		else throw runtime_error((string("unknown key: ") + argv[i]));
	}

	// Check arguments
	if(arguments.maxNumKpts < 0)
		throw runtime_error(string("ERROR: maximum number of keypoints must be > 0\n"));	
	else if(arguments.maxNumKpts > 5000)
		throw runtime_error(string("ERROR: maximum number of keypoints should not exceed 5000\n"));

	if(!arguments.procWidth == 0)	// not set
	{
		if(arguments.procWidth < 100)
			throw runtime_error(string("ERROR: too low procWidth (should be > 100 && < 1000)\n"));
		else if(arguments.procWidth > 2000)
			throw runtime_error(string("ERROR: too large procWidth (should be > 100 && < 1000)\n"));
	}

	return arguments;
}
// Application construction (initialization)
Application::Application(const Arguments &s)
{
	// Set arguments
	arguments = s;

	
	cout << "\nControls:\n"
      << "\tESC - exit\n"     
//        << "\tg - convert image to gray or not\n"
      << "\t1/q - increase/decrease maximum number kpts\n"
//        << "\t2/w - increase/decrease levels count\n"
//        << "\t3/e - increase/decrease HOG group threshold\n"
//        << "\t4/r - increase/decrease hit threshold\n"
        << endl;

	// Set from arguments
	if(arguments.videoFileName != 0)
		useCamera = false;

	// Default values
	fourccOut = CV_FOURCC('D', 'I', 'V', 'X');
	
}
void Application::run()
{
	// Video capture and writer
	cv::VideoCapture video;
	cv::VideoWriter videoOut;

	// Images
	cv::Mat inputImg, outputImg;	
	cv::Mat outputMosaic;

	// Open video input	
	if( useCamera )
		video.open(0);
	else	
		video.open(arguments.videoFileName);

	// Check video input
	int width = 0, height = 0, fps = 0, fourcc = 0;
	if( !video.isOpened() )	
		throw runtime_error(string("ERROR: can not open camera or video file\n"));	
	else
	{
		// Show video information
		width = (int) video.get(CV_CAP_PROP_FRAME_WIDTH);
		height = (int) video.get(CV_CAP_PROP_FRAME_HEIGHT);
		fps = (int) video.get(CV_CAP_PROP_FPS);
		fourcc = (int) video.get(CV_CAP_PROP_FOURCC);
		if(!useCamera)
			printf("Input video: (%d x %d) at %d fps, fourcc = %d\n", width, height, fps, fourcc);
		else
			printf("Input camera: (%d x %d) at %d fps\n", width, height, fps);
	}	

	// Resize	
	if(arguments.procWidth != 0)
	{	
		int procHeight = (int)(height*((double)arguments.procWidth/width));
		procSize = cv::Size(arguments.procWidth, procHeight);

		printf("Resize to: (%d x %d)\n", procSize.width, procSize.height);	
	}
	else
		procSize = cv::Size(width, height);

	// Video mosaic	
	outputMosaic = cv::Mat(cv::Size(2*procSize.width,2*procSize.height), CV_8UC3);	
		           
	// Output video(s)
	//videoFileNameOut = std::string("output_") + std::string(arguments.videoFileName);
	videoFileNameOut = "output.avi";
	
	if(arguments.saveVideoOut)
	{
		
		// Append mosaic vertically
		videoOut.open(videoFileNameOut, fourccOut, 25, outputMosaic.size());			
		
		
		if(!videoOut.isOpened())
		{
			throw runtime_error(string("ERROR: can not create video %s\n", videoFileNameOut.c_str()));			
		}
		else
			printf("Saving output video %s\n", videoFileNameOut.c_str());
	}
		
	// ---------------------------
	// Create/Init Variables
	// SIFT...
	SIFT_detector = new cv::SiftFeatureDetector();
	SIFT_descriptor = new cv::SiftDescriptorExtractor();
	SIFT_matcher = DescriptorMatcher::create("BruteForce");

	// ORB...
	ORB_detector = new cv::OrbFeatureDetector();
//	ORB_detector = new cv::GridAdaptedFeatureDetector(new ORB(10, true), arguments.maxNumKpts, 4, 4);
	ORB_descriptor = new cv::OrbDescriptorExtractor();
	ORB_matcher = DescriptorMatcher::create("BruteForce-Hamming");

	// FAST
	FAST_detector = new cv::GridAdaptedFeatureDetector(new FastFeatureDetector(10, true), arguments.maxNumKpts, 4, 4);
	FAST_descriptor = new cv::BriefDescriptorExtractor(32);
	FAST_matcher = DescriptorMatcher::create("BruteForce-Hamming");

	// FFME...
	ffme.init(procSize, arguments.maxNumKpts);

	// ---------------------------

	// MAIN LOOP
	printf("Model created...\n");
	int frameNum=0;
	for( ;; )
	{	
		// Get current image		
		video >> inputImg;			
	
		if( inputImg.empty() )	break;
		if( arguments.verbose ) printf("\nFRAME %6d - ", frameNum); fflush(stdout);
				
		// Resize to processing size
		cv::resize(inputImg, inputImg, procSize);		

		// Color Conversion to output (for video)
		if(inputImg.channels() == 3)	inputImg.copyTo(outputImg);				
		else	cv::cvtColor(inputImg, outputImg, CV_GRAY2BGR);

				
		// ++++++++++++++++++++++++++++++++++++++++
		// Process
		workBegin();
		work(inputImg, frameNum);				
		
		// Draw output image
		cv::rectangle(SIFT_outputImg, cv::Rect(0,0, 200, 32), CV_RGB(0,0,0), -1);
		cv::rectangle(FFME_outputImg, cv::Rect(0,0, 200, 32), CV_RGB(0,0,0), -1);
		cv::rectangle(ORB_outputImg, cv::Rect(0,0, 200, 32), CV_RGB(0,0,0), -1);
		cv::rectangle(FAST_outputImg, cv::Rect(0,0, 200, 32), CV_RGB(0,0,0), -1);
		cv::putText(SIFT_outputImg, "SIFT: " + siftWorkFps(), Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		cv::putText(ORB_outputImg, "ORB: " + orbWorkFps(), Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		cv::putText(FFME_outputImg, "FFME: " + ffmeWorkFps(), Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		cv::putText(FAST_outputImg, "FAST: " + fastWorkFps(), Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));

		stringstream ss;
		ss << SIFT_matches.size() << " matches";
		cv::putText(SIFT_outputImg, ss.str(), Point(40, 25), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		ss.str(std::string());	// to clear the stringstream
		ss << FFME_matches.size() << " matches";
		cv::putText(FFME_outputImg, ss.str(), Point(40, 25), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		ss.str(std::string());
		ss << ORB_matches.size() << " matches";
		cv::putText(ORB_outputImg, ss.str(), Point(40, 25), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
		ss.str(std::string());
		ss << FAST_matches.size() << " matches";
		cv::putText(FAST_outputImg, ss.str(), Point(40, 25), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0,255,0));
        
		// Create mosaic
		outputMosaic.setTo(0);
		
		// Append vertically
		cv::Rect rect1 = Rect(0,0,procSize.width, procSize.height);
		cv::Mat roi1 = outputMosaic(rect1);
		FFME_outputImg.copyTo(roi1);

		cv::Rect rect2 = Rect(procSize.width, 0, procSize.width, procSize.height);
		cv::Mat roi2 = outputMosaic(rect2);
		SIFT_outputImg.copyTo(roi2);

		cv::Rect rect3 = Rect(0, procSize.height, procSize.width, procSize.height);
		cv::Mat roi3 = outputMosaic(rect3);
		ORB_outputImg.copyTo(roi3);

		cv::Rect rect4 = Rect(procSize.width, procSize.height, procSize.width, procSize.height);
		cv::Mat roi4 = outputMosaic(rect4);
		FAST_outputImg.copyTo(roi4);
				
		// View
		imshow("Comparison window", outputMosaic);

		// Store video
		if(videoOut.isOpened() && arguments.saveVideoOut) videoOut << outputMosaic;//outputImg;

		workEnd();	
	
		// ++++++++++++++++++++++++++++++++++++++++
		// Handle key
		handleKey((char) waitKey(3));

		frameNum++;
	}
}

int Application::work(cv::Mat &inputImg, int frameNum)
{
	cv::Mat imGray;
	if(inputImg.channels() != 3)
		cvtColor(inputImg, imGray, CV_RGB2GRAY);
	else
		inputImg.copyTo(imGray);

	// SIFT...
	siftWorkBegin();
	inputImg.copyTo(SIFT_outputImg);
	SIFT_detector->detect(imGray, SIFT_query_kpts);
	SIFT_descriptor->compute(imGray, SIFT_query_kpts, SIFT_query_desc);
	if(SIFT_H_prev.empty())
		SIFT_H_prev = Mat::eye(3,3,CV_32FC1);

	std::vector<unsigned char> SIFT_match_mask;

	if(!SIFT_train_kpts.empty())
	{
		std::vector<cv::KeyPoint> test_kpts;
		warpKeypoints(SIFT_H_prev.inv(), SIFT_query_kpts, test_kpts);
		cv::Mat SIFT_mask = windowedMatchingMask(test_kpts, SIFT_train_kpts, 25, 25);
		SIFT_matcher->match(SIFT_query_desc, SIFT_train_desc, SIFT_matches, SIFT_mask);
		
		matches2points(SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_train_pts, SIFT_query_pts);
		
		
		if(SIFT_matches.size() > 5)
		{
			cv::Mat H = findHomography(SIFT_train_pts, SIFT_query_pts, RANSAC, 4, SIFT_match_mask);
			if(countNonZero(Mat(SIFT_match_mask)) > 15)
				SIFT_H_prev = H;
			else
				SIFT_H_prev = Mat::eye(3,3,CV_32FC1);

			drawMatchesRelative(SIFT_train_kpts, SIFT_query_kpts, SIFT_matches, SIFT_outputImg, SIFT_match_mask);
		}
	}
	else
	{	
		SIFT_H_prev = Mat::eye(3,3,CV_32FC1);
	}

	SIFT_train_kpts = SIFT_query_kpts;
	SIFT_query_desc.copyTo(SIFT_train_desc);	
	
	if(arguments.verbose)
		cout << "SIFT matches: " << SIFT_matches.size();

	siftWorkEnd();

	// ORB...
	orbWorkBegin();
	inputImg.copyTo(ORB_outputImg);
	ORB_detector->detect(imGray, ORB_query_kpts);
	ORB_descriptor->compute(imGray, ORB_query_kpts, ORB_query_desc);
	if(ORB_H_prev.empty())
		ORB_H_prev = Mat::eye(3,3,CV_32FC1);

	std::vector<unsigned char> ORB_match_mask;

	if(!ORB_train_kpts.empty())
	{
		std::vector<cv::KeyPoint> test_kpts;
		warpKeypoints(ORB_H_prev.inv(), ORB_query_kpts, test_kpts);
		cv::Mat ORB_mask = windowedMatchingMask(test_kpts, ORB_train_kpts, 25, 25);
		ORB_matcher->match(ORB_query_desc, ORB_train_desc, ORB_matches, ORB_mask);
		
		matches2points(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_train_pts, ORB_query_pts);
		
		
		if(ORB_matches.size() > 5)
		{
			cv::Mat H = findHomography(ORB_train_pts, ORB_query_pts, RANSAC, 4, ORB_match_mask);
			if(countNonZero(Mat(ORB_match_mask)) > 15)
				ORB_H_prev = H;
			else
				ORB_H_prev = Mat::eye(3,3,CV_32FC1);

			drawMatchesRelative(ORB_train_kpts, ORB_query_kpts, ORB_matches, ORB_outputImg, ORB_match_mask);
		}
	}
	else
	{	
		ORB_H_prev = Mat::eye(3,3,CV_32FC1);
	}

	ORB_train_kpts = ORB_query_kpts;
	ORB_query_desc.copyTo(ORB_train_desc);	
	
	if(arguments.verbose)
		cout << ", ORB matches: " << ORB_matches.size();

	orbWorkEnd();

	// FAST...
	fastWorkBegin();
	inputImg.copyTo(FAST_outputImg);
	FAST_detector->detect(imGray, FAST_query_kpts);
	FAST_descriptor->compute(imGray, FAST_query_kpts, FAST_query_desc);
	if(FAST_H_prev.empty())
		FAST_H_prev = Mat::eye(3,3,CV_32FC1);

	std::vector<unsigned char> FAST_match_mask;

	if(!FAST_train_kpts.empty())
	{
		std::vector<cv::KeyPoint> test_kpts;
		warpKeypoints(FAST_H_prev.inv(), FAST_query_kpts, test_kpts);
		cv::Mat FAST_mask = windowedMatchingMask(test_kpts, FAST_train_kpts, 25, 25);
		FAST_matcher->match(FAST_query_desc, FAST_train_desc, FAST_matches, FAST_mask);
		
		matches2points(FAST_train_kpts, FAST_query_kpts, FAST_matches, FAST_train_pts, FAST_query_pts);
		
		
		if(FAST_matches.size() > 5)
		{
			cv::Mat H = findHomography(FAST_train_pts, FAST_query_pts, RANSAC, 4, FAST_match_mask);
			if(countNonZero(Mat(FAST_match_mask)) > 15)
				FAST_H_prev = H;
			else
				FAST_H_prev = Mat::eye(3,3,CV_32FC1);

			drawMatchesRelative(FAST_train_kpts, FAST_query_kpts, FAST_matches, FAST_outputImg, FAST_match_mask);
		}
	}
	else
	{	
		FAST_H_prev = Mat::eye(3,3,CV_32FC1);
	}

	FAST_train_kpts = FAST_query_kpts;
	FAST_query_desc.copyTo(FAST_train_desc);	
	
	if(arguments.verbose)
		cout << ", FAST matches: " << FAST_matches.size();

	fastWorkEnd();

	// FFME...
	ffmeWorkBegin();
	inputImg.copyTo(FFME_outputImg);
	ffme.detect(imGray, FFME_query_kpts);
	ffme.describe(imGray, FFME_query_kpts, FFME_query_desc);	

	if(ffme.started())
	{
		ffme.match(FFME_query_kpts, FFME_train_kpts, FFME_matches);
		drawMatchesRelative(FFME_train_kpts, FFME_query_kpts, FFME_matches, FFME_outputImg);		  
	}
	ffme.updateBuffers();		

	if(arguments.verbose)
		cout << ", FFME matches: " << FFME_matches.size();


	ffmeWorkEnd();

	return 0;
}

void Application::handleKey(char key)
{
    switch (key)
    {
    case 27:
    //  running = false;
	exit(1);
        break;
    case '1':
        arguments.maxNumKpts = (int) arguments.maxNumKpts*1.05;
        cout << "maxNumKpts: " << arguments.maxNumKpts << endl;
        break;
    case 'q':
    case 'Q':
        arguments.maxNumKpts = (int) arguments.maxNumKpts/1.05;
        cout << "maxNumKpts: " << arguments.maxNumKpts << endl;
        break;
/*    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;*/

    }
}

// FFME time control
inline void Application::ffmeWorkBegin() { ffme_work_begin = getTickCount(); }

inline void Application::ffmeWorkEnd()
{
    int64 delta = getTickCount() - ffme_work_begin;
    double freq = getTickFrequency();
    ffme_work_fps = freq / delta;
    ffme_work_ms = (int64)(1000/ffme_work_fps);
}

inline string Application::ffmeWorkFps() const
{
    stringstream ss;
    ss << ffme_work_fps << "(fps) / " << ffme_work_ms << "(ms)";
    return ss.str();
}


// SIFT time control
inline void Application::siftWorkBegin() { sift_work_begin = getTickCount(); }

inline void Application::siftWorkEnd()
{
    int64 delta = getTickCount() - sift_work_begin;
    double freq = getTickFrequency();
    sift_work_fps = freq / delta;
    sift_work_ms = (int64)(1000/sift_work_fps);
}

inline string Application::siftWorkFps() const
{
    stringstream ss;
    ss << sift_work_fps << "(fps) / " << sift_work_ms << "(ms)";
    return ss.str();
}

// ORB time control
inline void Application::orbWorkBegin() { orb_work_begin = getTickCount(); }

inline void Application::orbWorkEnd()
{
    int64 delta = getTickCount() - orb_work_begin;
    double freq = getTickFrequency();
    orb_work_fps = freq / delta;
    orb_work_ms = (int64)(1000/orb_work_fps);
}

inline string Application::orbWorkFps() const
{
    stringstream ss;
    ss << orb_work_fps << "(fps) / " << orb_work_ms << "(ms)";
    return ss.str();
}

// FAST time control
inline void Application::fastWorkBegin() { fast_work_begin = getTickCount(); }

inline void Application::fastWorkEnd()
{
    int64 delta = getTickCount() - fast_work_begin;
    double freq = getTickFrequency();
    fast_work_fps = freq / delta;
    fast_work_ms = (int64)(1000/fast_work_fps);
}

inline string Application::fastWorkFps() const
{
    stringstream ss;
    ss << fast_work_fps << "(fps) / " << fast_work_ms << "(ms)";
    return ss.str();
}

// App time control
inline void Application::workBegin() { work_begin = getTickCount(); }

inline void Application::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string Application::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
