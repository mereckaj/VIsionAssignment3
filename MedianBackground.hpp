//
// Created by mereckaj on 12/12/15.
//

#ifndef VISIONASSIGNMENT3_MEDIANBACKGROUND_HPP
#define VISIONASSIGNMENT3_MEDIANBACKGROUND_HPP

#include <iostream>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class MedianBackground {
private:
    Mat mMedianBackground;
    float ****mHistogram;
    float ***mLessThanMedian;
    float mAgingRate;
    float mCurrentAge;
    float mTotalAges;
    int mValuesPerBin;
    int mNumberOfBins;
public:
    MedianBackground(Mat initial_image, float aging_rate, int values_per_bin);

    Mat GetBackgroundImage();

    void UpdateBackground(Mat current_frame);

    float getAgingRate() {
        return mAgingRate;
    }
};

#endif //VISIONASSIGNMENT3_MEDIANBACKGROUND_HPP
