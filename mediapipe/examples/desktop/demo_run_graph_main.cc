// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sys/stat.h>
#include <tuple>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

// for BlazePose
constexpr char kOutputLandmark[] = "pose_landmarks";

//for MHT
constexpr char kOutputDetections[] = "multi_palm_detections";
constexpr char kOutputLandmarks[] = "multi_hand_landmarks";
constexpr char kOutputPalmRects[] = "multi_palm_rects";
constexpr char kOutputHandRects[] = "multi_hand_rects";
constexpr char kOutputLandmarksRaw[] = "multi_hand_landmarks_raw";
constexpr char kOutputHandednesses[] = "handedness";
constexpr char kOutputHandPresenceScore[] = "multi_hand_presence_score";

typedef std::string OutputStreamName;
typedef std::string FilepathName;
typedef std::tuple<OutputStreamName, FilepathName,
                   std::function<bool(mediapipe::OutputStreamPoller *, int,
                                      FilepathName, std::string)>> HandTrackingDataInfo;

template <typename T>
bool extractHandTrackingDataToFile(mediapipe::OutputStreamPoller *poller,
                               int iLoop, FilepathName filepath,
                               std::string output_dirpath) {
  //std::cout << "check" << std::endl;
  if (poller->QueueSize() > 0) {
    mediapipe::Packet packet;
    //std::cout << "check packet" << std::endl;
    if (!poller->Next(&packet)){
      return false;
    }

    //std::cout << "get" << std::endl;
    auto &output_detections = packet.Get<std::vector<T>>();

    // output file
    //std::cout << "output file" << std::endl;
    for (int j = 0; j < output_detections.size(); ++j) {
      std::ostringstream os;
      os << output_dirpath + "/"
         << "iLoop=" << iLoop << "_" << filepath << "_"
         << "j=" << j << ".txt";
      std::ofstream outputfile(os.str());

      std::string serializedStr;
      output_detections[j].SerializeToString(&serializedStr);
      outputfile << serializedStr << std::flush;
    }
  }
  return true;
}

const std::map<std::string, HandTrackingDataInfo> HAND_TRACKING_DATA_INFO{
  {
    "detection",
    {
      kOutputDetections, "detection",
      extractHandTrackingDataToFile<mediapipe::Detection>
    }
  },
  {
    "landmark",
    {
      kOutputLandmarks, "landmark",
      extractHandTrackingDataToFile<mediapipe::NormalizedLandmarkList>
    }
  },
  {
    "palm_rect",
    {
      kOutputPalmRects, "palmRect",
      extractHandTrackingDataToFile<mediapipe::NormalizedRect>
    }
  },
  {
    "hand_rect",
    {
      kOutputHandRects, "handRect",
      extractHandTrackingDataToFile<mediapipe::NormalizedRect>
    }
  },
  {
    "landmark_raw",
    {
      kOutputLandmarksRaw, "landmarkRaw",
      extractHandTrackingDataToFile<mediapipe::LandmarkList>
    }
  },
  {
    "handedness",
    {
      kOutputHandednesses, "handedness",
      extractHandTrackingDataToFile<mediapipe::ClassificationList>
    }
  },
  // {
  //   "hand_presence_score",
  //   {
  //     kOutputHandPresenceScore, "presenceScore",
  //     extractHandTrackingDataToFile<float>
  //   }
  // },
};

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

cv::Mat executeGraph(cv::Mat input_image_raw, mediapipe::CalculatorGraph *graph,
                     mediapipe::OutputStreamPoller *poller_video,
                     std::map<std::string, mediapipe::OutputStreamPoller *> pollers,
                     std::string output_dirpath, int iLoop) {
  // save input frame to file
  std::ostringstream osInputFrame;
  osInputFrame << output_dirpath + "/"
               << "iLoop=" << iLoop << "_" 
               << "inputFrame"
               << ".jpg";
  cv::imwrite(osInputFrame.str(), input_image_raw);

  // convert color BGR to RGB
  cv::Mat input_image;
  cv::cvtColor(input_image_raw, input_image, cv::COLOR_BGR2RGB);

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, input_image.cols, input_image.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input_image.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  graph->AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));
  
  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet_video;
  if (!(poller_video->Next(&packet_video))) {
    return cv::Mat();
  }

  // get output from graph
  auto &output_video = packet_video.Get<mediapipe::ImageFrame>();

  // show output image
  // Convert back to opencv for display or saving.
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_video);
  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

  // save output frame to file
  std::ostringstream osOutputFrame;
  osOutputFrame << output_dirpath + "/"
                << "iLoop=" << iLoop << "_outputFrame.jpg";
  cv::imwrite(osOutputFrame.str(), output_frame_mat);

  // extract hand-tracking-infos from the graph to files
  for (auto const &[k, v] : HAND_TRACKING_DATA_INFO) {
    auto &stream_name = std::get<0>(v);
    auto &filepath = std::get<1>(v);
    auto &func = std::get<2>(v);
    auto poller = pollers[stream_name];

    func(poller, iLoop, filepath, output_dirpath);
  }

  return output_frame_mat;
}

void RunHands(
  mediapipe::CalculatorGraph *graph,
  mediapipe::OutputStreamPoller *poller_video,
  std::map<std::string, mediapipe::OutputStreamPoller *> pollers,
  std::string FLAGS_input_video_path, std::string output_dirpath){

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  if (!capture.isOpened()) {
    return;
  };

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  } else {
    LOG(INFO) << "Prepare video writer.";
    writer.open(FLAGS_output_video_path,
               mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
               capture.get(cv::CAP_PROP_FPS), 
               cv::Size(capture.get(cv::CAP_PROP_FRAME_HEIGHT),
                        capture.get(cv::CAP_PROP_FRAME_WIDTH)));
    if (!writer.isOpened()){
          return;
    }
  }

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  int iLoop = 0;

  while (grab_frames) {
    std::cout << "iLoop: " << iLoop
              << "-------------------------------------------------------------"
              << std::endl;
    std::cerr << "iLoop: " << iLoop
              << "-------------------------------------------------------------"
              << std::endl;
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty())
      break; // End of video.
    if (!load_video) {
      cv::flip(camera_frame_raw, camera_frame_raw, /*flipcode=HORIZONTAL*/ 1);
    }
    auto output_frame_mat = executeGraph(camera_frame_raw, graph, poller_video,
                                         pollers, output_dirpath, iLoop);
    if (save_video) {
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
    std::cout << std::endl;
    ++iLoop;
  }
  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
}

// ::mediapipe::Status RunMPPGraph() {
//   std::string calculator_graph_config_contents;
//   MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
//       FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
//   LOG(INFO) << "Get calculator graph config contents: "
//             << calculator_graph_config_contents;
//   mediapipe::CalculatorGraphConfig config =
//       mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
//           calculator_graph_config_contents);

//   LOG(INFO) << "Initialize the calculator graph.";
//   mediapipe::CalculatorGraph graph;
//   MP_RETURN_IF_ERROR(graph.Initialize(config));

//   // Connect pollers to graph
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video,
//                    graph.AddOutputStreamPoller(kOutputStream));
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_detections,
//                    graph.AddOutputStreamPoller(kOutputDetections));
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks,
//                    graph.AddOutputStreamPoller(kOutputLandmarks));
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_palm_rects,
//                    graph.AddOutputStreamPoller(kOutputPalmRects));
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_hand_rects,
//                    graph.AddOutputStreamPoller(kOutputHandRects));
//   ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_handednesses,
//                    graph.AddOutputStreamPoller(kOutputHandednesses));

//   std::map<std::string, mediapipe::OutputStreamPoller *> pollers;
//   pollers[kOutputDetections] = &poller_detections;
//   pollers[kOutputLandmarks] = &poller_landmarks;
//   pollers[kOutputPalmRects] = &poller_palm_rects;
//   pollers[kOutputHandRects] = &poller_hand_rects;
//   pollers[kOutputHandednesses] = &poller_handednesses;

//   // define output foler path
//   std::string output_filename = FLAGS_output_video_path.substr(FLAGS_output_video_path.rfind("/") + 1);
//   std::string output_dirpath = std::string() + "./result/MHT/" + output_filename + "/";
//   mkdir(output_dirpath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

//   LOG(INFO) << "Start running the calculator graph.";

//   MP_RETURN_IF_ERROR(graph.StartRun({}));

//   RunHands(&graph, &poller_video, pollers, FLAGS_input_video_path, output_dirpath);

//   MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
//   return graph.WaitUntilDone();
// }

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));

  //for BlazePose
  //------------
  // ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_pose_landmarks,
  //                  graph.AddOutputStreamPoller(kOutputLandmark));
  //------------

  //for MHT
  //------------
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_detections,
                   graph.AddOutputStreamPoller(kOutputDetections));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks,
                   graph.AddOutputStreamPoller(kOutputLandmarks));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_palm_rects,
                   graph.AddOutputStreamPoller(kOutputPalmRects));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_hand_rects,
                   graph.AddOutputStreamPoller(kOutputHandRects));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_raw,
                   graph.AddOutputStreamPoller(kOutputLandmarksRaw));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_handednesses,
                   graph.AddOutputStreamPoller(kOutputHandednesses));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_presence_score,
                   graph.AddOutputStreamPoller(kOutputHandPresenceScore));

  std::map<std::string, mediapipe::OutputStreamPoller *> pollers;
  pollers[kOutputDetections] = &poller_detections;
  pollers[kOutputLandmarks] = &poller_landmarks;
  pollers[kOutputPalmRects] = &poller_palm_rects;
  pollers[kOutputHandRects] = &poller_hand_rects;
  pollers[kOutputLandmarksRaw] = &poller_landmarks_raw;
  pollers[kOutputHandednesses] = &poller_handednesses;
 //pollers[kOutputHandPresenceScore] = &poller_presence_score;
  //----------------
  
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  int iLoop = 0;
  // define output folder path
  std::string output_filename = FLAGS_output_video_path.substr(FLAGS_output_video_path.rfind("/") + 1);
  std::string output_dirpath;
  // for BlazePose
  //output_dirpath = std::string() + "./result/blazepose/" + output_filename + "/";

  // for MHT
  output_dirpath = std::string() + "./result/MHT/" + output_filename + "/";

  mkdir(output_dirpath.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

  while (grab_frames) {
    std::cout << "iLoop: " << iLoop 
              << "-------------------------------------------------------------------" 
              << std::endl;
    std::cerr << "iLoop: " << iLoop 
              << "--------------------------------------------------------------------" 
              << std::endl;
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet_video;
    if (poller.Next(&packet_video)){
      auto& output_frame = packet_video.Get<mediapipe::ImageFrame>();

      cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
      if (save_video) {
        if (!writer.isOpened()) {
          LOG(INFO) << "Prepare video writer.";
          writer.open(FLAGS_output_video_path,
                      mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                      capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
          RET_CHECK(writer.isOpened());
        }
        writer.write(output_frame_mat);
      } else {
        cv::imshow(kWindowName, output_frame_mat);
        // Press any key to exit.
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
      }
    // save output frame to file
      std::cout << "save output frame to file" << std::endl;

      std::ostringstream osOutputFrame;
      osOutputFrame << output_dirpath + "/"
                    << "iLoop=" << iLoop
                    << "_outputFrame.jpg";
      cv::imwrite(osOutputFrame.str(), output_frame_mat);
    } else {
      break;
    }

    // for BlazePose
    // -------------------
    // if (poller_pose_landmarks.QueueSize() > 0){
    //   mediapipe::Packet packet_landmarks;
    //   if (!poller_pose_landmarks.Next(&packet_landmarks)) break;

    //   auto& pose_landmarks = packet_landmarks.Get<mediapipe::NormalizedLandmarkList>();

    //   //Output file
    //   std::cout << "Output file" << std::endl;
    //   std::ostringstream os;
    //   os << output_dirpath + "/"
    //     << "iLoop=" << iLoop << "_"
    //     << "landmark" << ".txt";
    //   std::ofstream outputfile(os.str());
    
    //   std::string serializedStr;
    //   pose_landmarks.SerializeToString(&serializedStr);
    //   outputfile << serializedStr << std::flush;
    // }
    //----------------
  
    //for MHT
    //--------------
    // extract hand-tracking-infos from the graph to files
    std::cout << "extract hand-tracking-infos from the graph to files" << std::endl;
    for (auto const &[k, v] : HAND_TRACKING_DATA_INFO) {
      auto &stream_name = std::get<0>(v);
      auto &filepath = std::get<1>(v);
      auto &func = std::get<2>(v);
      auto poller = pollers[stream_name];

      //std::cout << filepath << std::endl;
      func(poller, iLoop, filepath, output_dirpath);
    }

    if (poller_presence_score.QueueSize() > 0) {
      mediapipe::Packet packet;
      //std::cout << "check packet" << std::endl;
      if (!poller_presence_score.Next(&packet)){
        break;
      }

      //std::cout << "get" << std::endl;
      auto &output_detections = packet.Get<std::vector<float>>();

      // output file
      //std::cout << "output file" << std::endl;
      for (int j = 0; j < output_detections.size(); ++j) {
        std::cout << "presence_score: " << output_detections[j] << std::endl;

        std::ostringstream os;
        os << output_dirpath + "/"
          << "iLoop=" << iLoop << "_" << "presenceScore" << "_"
          << "j=" << j << ".json";
        std::ofstream outputfile(os.str());

        std::ostringstream ss;
        ss << "{ \"presenceScore\": " << output_detections[j] << "}";
        outputfile << ss.str() << std::flush;
      }
    }
   // ---------------------    

    //Convert back to opencv for display or saving.
    // cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    // if (save_video) {
    //   if (!writer.isOpened()) {
    //     LOG(INFO) << "Prepare video writer.";
    //     writer.open(FLAGS_output_video_path,
    //                 mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
    //                 capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
    //     RET_CHECK(writer.isOpened());
    //   }
    //   writer.write(output_frame_mat);
    // } else {
    //   cv::imshow(kWindowName, output_frame_mat);
    //   // Press any key to exit.
    //   const int pressed_key = cv::waitKey(5);
    //   if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    // }

    // // save output frame to file
    // std::cout << "save output frame to file" << std::endl;

    // std::ostringstream osOutputFrame;
    // osOutputFrame << output_dirpath + "/"
    //               << "iLoop=" << iLoop
    //               << "_outputFrame.jpg";
    // cv::imwrite(osOutputFrame.str(), output_frame_mat);

    std::cout << std::endl;
    ++iLoop;
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
