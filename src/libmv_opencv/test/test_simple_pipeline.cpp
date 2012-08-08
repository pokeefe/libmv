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

#include "test_precomp.hpp"

#include "libmv/simple_pipeline/bundle.h"
#include "libmv/simple_pipeline/camera_intrinsics.h"
#include "libmv/simple_pipeline/initialize_reconstruction.h"
#include "libmv/simple_pipeline/pipeline.h"
#include "libmv/simple_pipeline/tracks.h"

#include <fstream>
#include <cstdlib>

using namespace cv;
using namespace std;

/**
 * 2D tracked points (dinosaur dataset)
 * ------------------------------------
 *
 * The format is:
 *
 * row1 : x1 y1 x2 y2 ... x36 y36 for track 1
 * row2 : x1 y1 x2 y2 ... x36 y36 for track 2
 * etc
 *
 * i.e. a row gives the 2D measured position of a point as it is tracked
 * through frames 1 to 36.  If there is no match found in a view then x
 * and y are -1.
 *
 * Each row corresponds to a different point.
 *
 * Link: http://www.robots.ox.ac.uk/~vgg/data/data-mview.html
 */
void vgg_2D_tracked_points_parser( libmv::Tracks &libmv_tracks )
{
    string filename = string(TEST_DATA_DIR) + "viff.xy.good_tracks.txt";
    ifstream file( filename.c_str() );

    const int height = 576;

    double x, y;
    string str;

    for (int track = 0; getline(file, str); ++track)
    {
        istringstream line(str);
        bool is_first_time;

        for (int frame = 0; line >> x >> y; ++frame)
        {
            // init track
            if ( is_first_time && x > 0 && y > 0 )
            {
                y = height - y;                               // for blender: x = x/720;  y = (576-y)/576;
                libmv_tracks.Insert( frame, track, x, y );
                is_first_time = false;
            }

            // while tracking
            else if ( x > 0 && y > 0 )
            {
                y = height - y;
                libmv_tracks.Insert( frame, track, x, y );
            }

            // lost track
            else if ( x < 0 && y < 0 )
            {
                is_first_time = true;
            }

            // some error
            else
            {
                exit(1);
            }
        }
    }
}



typedef struct libmv_Reconstruction
{
    libmv::EuclideanReconstruction reconstruction;

    /* used for per-track average error calculation after reconstruction */
    libmv::Tracks tracks;
    libmv::CameraIntrinsics intrinsics;

    double error;
} libmv_Reconstruction;


// ToDo (pablo): rewrite this, and move to "src/" folder
// Based on the 'libmv_capi' function (blender API)
void libmv_solveReconstruction(const libmv::Tracks &tracks, int keyframe1, int keyframe2,
                               double focal_length, double principal_x, double principal_y, double k1, double k2, double k3,
                               libmv_Reconstruction &libmv_reconstruction, bool refine_intrinsics = false)
{
    /* Invert the camera intrinsics. */
    libmv::vector<libmv::Marker> markers = tracks.AllMarkers();
    libmv::EuclideanReconstruction *reconstruction = &libmv_reconstruction.reconstruction;
    libmv::CameraIntrinsics *intrinsics = &libmv_reconstruction.intrinsics;

    intrinsics->SetFocalLength(focal_length, focal_length);
    intrinsics->SetPrincipalPoint(principal_x, principal_y);
    intrinsics->SetRadialDistortion(k1, k2, k3);

    cout << "\tNumber of markers: " << markers.size() << endl;
    for (int i = 0; i < markers.size(); ++i)
    {
        intrinsics->InvertIntrinsics(markers[i].x,
                                     markers[i].y,
                                     &(markers[i].x),
                                     &(markers[i].y));
    }

    libmv::Tracks normalized_tracks(markers);

    cout << "\tframes to init from: " << keyframe1 << " " << keyframe2 << endl;
    libmv::vector<libmv::Marker> keyframe_markers =
        normalized_tracks.MarkersForTracksInBothImages(keyframe1, keyframe2);
    cout << "\tNumber of markers for init: " << keyframe_markers.size() << endl;

    libmv::EuclideanReconstructTwoFrames(keyframe_markers, reconstruction);
    libmv::EuclideanBundle(normalized_tracks, reconstruction);
    libmv::EuclideanCompleteReconstruction(normalized_tracks, reconstruction);

    // ToDo (pablo): autocalibration?
//     if (refine_intrinsics) {
//         libmv_solveRefineIntrinsics(tracks, intrinsics, reconstruction,
//             refine_intrinsics, progress_update_callback, callback_customdata);
//     }

    libmv_reconstruction.tracks = tracks;
    libmv_reconstruction.error = libmv::EuclideanReprojectionError(tracks, *reconstruction, *intrinsics);
}



TEST(Sfm_simple_pipeline, dinosaur)
{
    libmv::Tracks tracks;
    vgg_2D_tracked_points_parser( tracks );


    int keyframe1 = 1, keyframe2 = 6;
    double focal_length = 24, principal_x = 360, principal_y = 288, k1 = 0, k2 = 0, k3 = 0;
    libmv_Reconstruction libmv_reconstruction;

    libmv_solveReconstruction( tracks, keyframe1, keyframe2,
                               focal_length, principal_x, principal_y, k1, k2, k3,
                               libmv_reconstruction );

    cout << "libmv_reconstruction.error = " << libmv_reconstruction.error << endl;

    // ToDo: complete the test
//     FAIL();
}