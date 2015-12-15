#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Utils.hpp"
#include "MedianBackground.hpp"

#define FPS 25
#define MS_PER_SEC 1000
#define FRAME_SKIP 5
#define INTER_FRAME_DELAY 5
//#define INTER_FRAME_DELAY (MS_PER_SEC / FPS )

const std::vector<std::string> fileNames = {
        "ObjectAbandonmentAndRemoval1.avi",
        "ObjectAbandonmentAndRemoval2.avi"
};
const std::string videoFileLocation("/home/mereckaj/Dev/ClionProjects/VisionAssignment3/Video/");

void ShowVideo(cv::VideoCapture capture);

int main(int argc, char **argv) {
    cv::VideoCapture *videos = LoadVideos(fileNames, videoFileLocation);
    for (int i = 0; i < fileNames.size(); i++) {
        ShowVideo(videos[i]);
    }
    return 0;
}

void GetDiff(cv::Mat median_background_image, cv::Mat current_frame, cv::Mat *median_difference) {
    cv::absdiff(median_background_image, current_frame, *median_difference);
    cv::cvtColor(*median_difference, *median_difference, CV_BGR2GRAY);
    cv::threshold(*median_difference, *median_difference, 30, 255, THRESH_BINARY);
}

std::vector<std::vector<cv::Point>> BinaryToContours(cv::Mat binaryImage) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    return contours;
}

std::vector<cv::Point> GetCentres(std::vector<std::vector<cv::Point>> contours) {

    // Get moments
    std::vector<cv::Moments> moments(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        moments[i] = cv::moments(contours[i], false);
    }

    // Get centres
    std::vector<cv::Point> mc;
    for (int i = 0; i < contours.size(); i++) {
        if (moments[i].m00 > 25) {
            mc.push_back(cv::Point((int) (moments[i].m10 / moments[i].m00), (int) (moments[i].m01 / moments[i].m00)));
        }
    }

    return mc;
}

void DrawCentres(std::vector<cv::Point> centres, cv::Mat *currentFrame) {
    for (size_t centre = 0; centre < centres.size(); centre++) {
        cv::circle(*currentFrame, centres[centre], 10, cv::Scalar(0, 255, 0));
    }
}

std::vector<cv::Point> GetSameCentres(std::vector<cv::Point> old, std::vector<cv::Point> neww) {
    std::vector<cv::Point> result;
    for (size_t oldCentre = 0; oldCentre < old.size(); oldCentre++) {
        for (size_t newCentre = 0; newCentre < neww.size(); newCentre++) {
            if (abs(old[oldCentre].x - neww[newCentre].x) < 5) {
                if (abs(old[oldCentre].y - neww[newCentre].y) < 5) {
                    result.push_back(neww[newCentre]);
                }
            }
        }
    }
    return result;
}

int BoundRectArea(cv::Rect rect) {
    cv::Point tl = rect.tl(), br = rect.br();
    int width = br.x - tl.x;
    int heght = br.y - tl.y;
    return width * heght;
}


bool CheckIfAbandonedOrRemoved(Mat cur, Mat old) {
    cv::Mat curEdges, oldEdge;
    int lowThrs = 1;
    Sobel(cur,curEdges,cur.depth(),0,1);
    Sobel(old,oldEdge,old.depth(),0,1);
    //TODO: histogram diffs
    convertScaleAbs(curEdges,curEdges);
    convertScaleAbs(oldEdge,oldEdge);
    int curEdgesCount = sum(curEdges)[0];
    int oldEdgesCount = sum(oldEdge)[0];
    cout << "Cur: " << curEdgesCount << endl;
    cout << "Old: " << oldEdgesCount << endl;
    cout << "Diff" << oldEdgesCount-curEdgesCount << endl;
    cv::waitKey(500);
    return false;
}

void ShowVideo(cv::VideoCapture capture) {
    cv::Mat currentFrame, medianBackgroundImageSlow, medianBackgroundImageFast, slowDiff,
            fastDiff, diff, tmp1, tmp2, tmpOriginal,originalBackground;
    int frameNumber = 0;
    int previousBiggestRectSize = 0;
    bool foundNewObject = true;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> centresNew, centresOld, centresToDraw;
    capture >> currentFrame;
    cv::blur(currentFrame, currentFrame, cv::Size(3, 3));
    currentFrame.copyTo(tmpOriginal);
    currentFrame.copyTo(originalBackground);
    MedianBackground medianBackgroundSlow(currentFrame, (float) 1.005, 1);
    MedianBackground medianBackgroundFast(currentFrame, (float) 1.05, 1);
    cv::Rect biggestRect,lastBiggestRect;
    while (!currentFrame.empty()) {
        if (frameNumber % FRAME_SKIP == 0) {
            medianBackgroundSlow.UpdateBackground(currentFrame);
            medianBackgroundFast.UpdateBackground(currentFrame);
            medianBackgroundImageSlow = medianBackgroundSlow.GetBackgroundImage();
            medianBackgroundImageFast = medianBackgroundFast.GetBackgroundImage();
            GetDiff(medianBackgroundImageSlow, currentFrame, &slowDiff);
            GetDiff(medianBackgroundImageFast, currentFrame, &fastDiff);
            cv::absdiff(slowDiff, fastDiff, diff);

            cv::morphologyEx(diff, tmp1, MORPH_OPEN, cv::Mat());
            cv::morphologyEx(tmp1, tmp2, MORPH_OPEN, cv::Mat());
            cv::morphologyEx(tmp2, tmp1, MORPH_CLOSE, cv::Mat());
            contours = BinaryToContours(tmp1);
            cv::drawContours(currentFrame, contours, -1, cv::Scalar(0, 0, 255));
            centresNew = GetCentres(contours);
            centresToDraw = GetSameCentres(centresOld, centresNew);

            if (centresToDraw.size() > 0) {
                vector<vector<Point> > contours_poly(contours.size());
                vector<Rect> boundRect(contours.size());
                int area = 0;

                for (int i = 0; i < contours.size(); i++) {
                    approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
                    boundRect[i] = boundingRect(Mat(contours_poly[i]));
                    if (BoundRectArea(boundRect[i]) > area) {
                        biggestRect = boundRect[i];
                    }
                }
                int val = BoundRectArea(biggestRect);
                if (previousBiggestRectSize == val && (biggestRect != lastBiggestRect)) {
                    lastBiggestRect = biggestRect;
                    //TODO: CHECK THAT NOT THE SAME OBJECT AS LAST FRAME
                    rectangle(currentFrame, biggestRect.tl(), biggestRect.br(), Scalar(0, 0, 255));
                    foundNewObject = true;
                    cout << "Found new object at: (" << biggestRect.tl().x << "," << biggestRect.tl().y << ")," <<
                    "(" << biggestRect.br().x << ", " << biggestRect.br().y << ")" << endl;
                    cv::Mat squareCurrent, square50ago;
                    squareCurrent = tmpOriginal(biggestRect);
                    square50ago = originalBackground(biggestRect);
                    bool result;
                    result = CheckIfAbandonedOrRemoved(squareCurrent, square50ago);
                    cv::imshow("Slow", currentFrame);
                    cv::moveWindow("Slow", 0, 0);
                } else if (previousBiggestRectSize < val) {
                    foundNewObject = false;
                } else {
                    cout << "Finding new object" << endl;
                }
                previousBiggestRectSize = val;
            }

            DrawCentres(centresToDraw, &currentFrame);
            cv::waitKey(INTER_FRAME_DELAY);
            diff = Mat::zeros(diff.size(), diff.type());
            centresOld = centresNew;
        }
        frameNumber++;
        capture >> currentFrame;
        cv::blur(currentFrame, currentFrame, cv::Size(3, 3));
        currentFrame.copyTo(tmpOriginal);
    }
}

