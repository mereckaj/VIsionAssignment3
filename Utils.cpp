//
// Created by mereckaj on 12/12/15.
//

#include "Utils.hpp"
#include <iostream>
#include <vector>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

cv::VideoCapture *LoadVideos(std::vector<std::string> video_files, std::string file_location) {
    using namespace std;
    cv::VideoCapture *videos = new cv::VideoCapture[video_files.size()];
    for (int video = 0; video < video_files.size(); video++) {
        string filename(file_location);
        filename.append(video_files[video]);
        videos[video].open(filename);
        if (!videos[video].isOpened()) {
            cout << "Cannot open video file: " << filename << endl;
            return NULL;
        } else {
            cout << "Sucessfully loaded: " << filename << endl;
        }

    }
    return videos;
}

/*
 * KDH's code
 */
Mat JoinImagesHorizontally( Mat& image1,Mat& image2, int spacing)
{
    Mat result( (image1.rows > image2.rows) ? image1.rows : image2.rows,
                image1.cols + image2.cols + spacing,
                image1.type() );
    result.setTo(Scalar(255,255,255));
    Mat imageROI;
    imageROI = result(cv::Rect(0,0,image1.cols,image1.rows));
    image1.copyTo(imageROI);
    if (spacing > 0)
    {
        imageROI = result(cv::Rect(image1.cols,0,spacing,image1.rows));
        imageROI.setTo(Scalar(255,255,255));
    }
    imageROI = result(cv::Rect(image1.cols+spacing,0,image2.cols,image2.rows));
    image2.copyTo(imageROI);
    return result;
}

Mat JoinImagesVertically( Mat& image1, Mat& image2, int spacing)
{
    Mat result( image1.rows + image2.rows + spacing,
                (image1.cols > image2.cols) ? image1.cols : image2.cols,
                image1.type() );
    result.setTo(Scalar(255,255,255));
    Mat imageROI;
    imageROI = result(cv::Rect(0,0,image1.cols,image1.rows));
    image1.copyTo(imageROI);
    if (spacing > 0)
    {
        imageROI = result(cv::Rect(0,image1.rows,image1.cols,spacing));
        imageROI.setTo(Scalar(255,255,255));
    }
    imageROI = result(cv::Rect(0,image1.rows+spacing,image2.cols,image2.rows));
    image2.copyTo(imageROI);
    return result;
}