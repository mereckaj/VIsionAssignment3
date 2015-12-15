//
// Created by mereckaj on 12/12/15.
//

#ifndef VISIONASSIGNMENT3_UTILS_HPP
#define VISIONASSIGNMENT3_UTILS_HPP

#include <vector>
#include <iostream>
#include <opencv2/videoio.hpp>

cv::Mat JoinImagesVertically(cv::Mat &image1, cv::Mat &image2, int spacing = 5);

cv::Mat JoinImagesHorizontally(cv::Mat &image1, cv::Mat &image2, int spacing = 5);


cv::VideoCapture *LoadVideos(std::vector<std::string> video_files, std::string file_location);
#endif //VISIONASSIGNMENT3_UTILS_HPP
