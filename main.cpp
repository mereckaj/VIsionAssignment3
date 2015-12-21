#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Utils.hpp"
#include "MedianBackground.hpp"

#define FPS 25
#define MS_PER_SEC 1000
#define FRAME_SKIP 5
#define INTER_FRAME_DELAY (MS_PER_SEC / FPS )

const std::vector<std::string> fileNames = {
        "ObjectAbandonmentAndRemoval1.avi",
        "ObjectAbandonmentAndRemoval2.avi"
};
const std::string videoFileLocation("/home/mereckaj/Dev/ClionProjects/VisionAssignment3/Video/");

const std::vector<std::vector<int>> groundTruth = {
        //fps,total frames,object becomes static, object picked up,obj tl x, obj tl y,obj br x, obj br y
        {25, 717, 183, 509, 356, 208, 390, 239},
        {25, 692, 215, 551, 287, 261, 352, 323}
};

std::vector<std::vector<Rect>> resultsEvent(2);


void ShowVideo(cv::VideoCapture capture, std::vector<int> t, std::vector<Rect> * event);

int main(int argc, char **argv) {
    cv::VideoCapture *videos = LoadVideos(fileNames, videoFileLocation);
    for (int i = 0; i < fileNames.size(); i++) {
        ShowVideo(videos[i], groundTruth[i],&resultsEvent[i]);
    }
    for (int i = 0; i < resultsEvent.size(); i++) {
        for(auto j = 0; j < resultsEvent[i].size();j++){
            cout << "((" << resultsEvent[i][j].tl().x << "," << resultsEvent[i][j].tl().y << ")";
            cout << "(" << resultsEvent[i][j].br().x << "," << resultsEvent[i][j].br().y << ")) ";
        }
        cout << endl;
    }

    return EXIT_SUCCESS;
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

void ShowVideo(cv::VideoCapture capture, std::vector<int> ground, std::vector<Rect> * event){
    cv::Mat currentFrame, medianBackgroundImageSlow, medianBackgroundImageFast, slowDiff,
            fastDiff, diff, tmp1, tmp2, tmpOriginal, originalBackground;
    int frameNumber = 0;
    int previousBiggestRectSize = 0;
    bool found = false, previousCheck = false;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> centresNew, centresOld, centresToDraw;

    /*
     * Get first frame and blur it to remove noise
     */
    capture >> currentFrame;
    cv::blur(currentFrame, currentFrame, cv::Size(3, 3));

    /*
     * Create a copy of the original frame for later use;
     */
    currentFrame.copyTo(tmpOriginal);
    currentFrame.copyTo(originalBackground);

    /*
     * Set up a fast and slow learning median background images
     */
    MedianBackground medianBackgroundSlow(currentFrame, (float) 1.005, 1);
    MedianBackground medianBackgroundFast(currentFrame, (float) 1.05, 1);
    cv::Rect biggestRect, previousBiggestRect;

    /*
     * Loop over every frame of the video
     */
    while (!currentFrame.empty()) {

        /*
         * Only process every FRAME_SKIP frame (5th)
         */
        if (frameNumber % FRAME_SKIP == 0) {

            /*
             * Update the median background with the current frame
             * Both fast and slow
             */
            medianBackgroundSlow.UpdateBackground(currentFrame);
            medianBackgroundFast.UpdateBackground(currentFrame);

            /*
             * Get the new median background images
             * Both fast and slow
             */
            medianBackgroundImageSlow = medianBackgroundSlow.GetBackgroundImage();
            medianBackgroundImageFast = medianBackgroundFast.GetBackgroundImage();

            /*
             * Get the difference between the current frame and the median background
             *
             */
            GetDiff(medianBackgroundImageSlow, currentFrame, &slowDiff);
            GetDiff(medianBackgroundImageFast, currentFrame, &fastDiff);

            /*
             * Get the differnece between the two difference images
             *
             * This should give me an image where one median background says there exists a non-background image
             * and the other one says it doesn't exist there
             */
            cv::absdiff(slowDiff, fastDiff, diff);

            /*
             * Morph operations to make sure the full bag is shown
             */
            cv::morphologyEx(diff, tmp1, MORPH_DILATE, cv::Mat());
            cv::morphologyEx(tmp1, tmp2, MORPH_OPEN, cv::Mat());
            cv::morphologyEx(tmp2, tmp1, MORPH_OPEN, cv::Mat());
            tmp1 = diff;
            /*
             * Get contours of all the objects in the image
             */
            contours = BinaryToContours(tmp1);
            /*
             * Get new centres from the contours and find centres which are the same as the previous frames centres
             * These will represent objects that have stayed the same over two frames
             */
            centresNew = GetCentres(contours);
            centresToDraw = GetSameCentres(centresOld, centresNew);

            /*
             * If there are more than 0 centres then draw them
             */
            if (centresToDraw.size() > 0 && !found) {

                /*
                 * Find the biggest bounding rectangle
                 */
                vector<vector<Point> > contours_poly(contours.size());
                vector<Rect> boundRect(contours.size());
                int area = 0;

                for (int i = 0; i < contours.size(); i++) {
                    approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
                    boundRect[i] = boundingRect(Mat(contours_poly[i]));
                    int boundRectArea = BoundRectArea(boundRect[i]);
                    if (boundRectArea > area) {
                        area = boundRectArea;
                        biggestRect = boundRect[i];
                    }
                }

                /*
                 * Get the area of the biggest rectangle
                 */
                if (previousBiggestRect == biggestRect) {
                    found = true;
                }
                previousBiggestRect = biggestRect;

            } else {
                if (previousCheck) {
                    found = false;
                    previousCheck = false;
                } else {
                    previousCheck = true;
                }
            }
            if (found) {
                Rect overlap = biggestRect & Rect(ground[4], ground[5], ground[6], ground[7]);
                double coeff = (double) (2 * overlap.area()) /
                               (double) (biggestRect.area() + Rect(ground[4], ground[5], ground[6], ground[7]).area());
                event->push_back(biggestRect);
                cout << "Coeff: " << coeff << endl;
//                cout << "FN: " << frameNumber << endl;
                rectangle(currentFrame, biggestRect.tl(), biggestRect.br(), cv::Scalar(0, 0, 255));
            }
            diff = Mat::zeros(diff.size(), diff.type());
            centresOld = centresNew;
        }
        frameNumber++;
        capture >> currentFrame;
        cv::blur(currentFrame, currentFrame, cv::Size(3, 3));
        currentFrame.copyTo(tmpOriginal);
    }
}
