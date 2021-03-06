/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <opencv2/sfm/conditioning.hpp>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/numeric.hpp>

#include <opencv2/core/eigen.hpp>

namespace cv
{

void
preconditionerFromPoints( const Mat_<double> &_points,
                          Matx33d &_Tr )
{

    Mat_<double> mean, variance;
    meanAndVarianceAlongRows(_points, mean, variance);

    double xFactor = sqrt(2.0 / variance.at<double>(0,0));
    double yFactor = sqrt(2.0 / variance.at<double>(1,0));

    // If variance is equal to 0.0 set scaling factor to identity.
    // -> Else it will provide nan value (because division by 0).
    if (variance.at<double>(0,0) < 1e-8)
        xFactor = mean.at<double>(0,0) = 1.0;
    
    if (variance.at<double>(1,0) < 1e-8)
        yFactor = mean.at<double>(1,0) = 1.0;

    _Tr << xFactor,       0, -xFactor*mean.at<double>(0,0),
                 0, yFactor, -yFactor*mean.at<double>(1,0),
                 0,       0,                             1;

}

void
isotropicPreconditionerFromPoints( const Mat &_points,
                                   Matx33d &_T )
{

    Mat_<double> mean, variance;
    meanAndVarianceAlongRows(_points, mean, variance);
    
    double var_norm = cv::norm(variance);
    double factor = sqrt(2.0 / var_norm);
    
    // If variance is equal to 0.0 set scaling factor to identity.
    // -> Else it will provide nan value (because division by 0).
    if (var_norm < 1e-8)
    {
        factor = 1.0;
        mean.setTo(1);
    }

    _T << factor,      0, -factor*mean.at<double>(0,0),
               0, factor, -factor*mean.at<double>(1,0),
               0,      0,                            1;

}

void
applyTransformationToPoints( const Mat &_points,
                             const Matx33d &_T,
                             Mat &_transformed_points )
{

    int numPoints = _points.cols;
    _transformed_points.create(2,numPoints, _points.type());
    Mat p(3, numPoints, _points.type());

    euclideanToHomogeneous(_points, p);
    p = Mat(_T) * p;
    homogeneousToEuclidean(p, _transformed_points);

}

void
normalizePoints( const Mat &points,
                 Mat &normalized_points,
                 Matx33d &T )
{
    preconditionerFromPoints(points, T);
    applyTransformationToPoints(points, T, normalized_points);
}

void
normalizeIsotropicPoints( const Mat &points,
                          Mat &normalized_points,
                          Matx33d &T )
{
    isotropicPreconditionerFromPoints(points, T);
    applyTransformationToPoints(points, T, normalized_points);
}

} /* namespace cv */
