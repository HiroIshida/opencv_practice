#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#define print( something ) std::cout<<something <<std::endl;

cv::Mat convert2gray(const cv::Mat& src_img);
cv::Mat gray_binarize(const cv::Mat& src_img);
std::vector<cv::Mat> three_functions(const cv::Mat& src_img);
cv::Mat face_detector(const cv::Mat& src_img);

int main(int argc, char **argv)
{
  // data is pointer type
  //http://opencv.jp/opencv-2svn/cpp/core_basic_structures.html#mat
  // about nullptr "Invalid pointers and null pointers"
  //http://www.cplusplus.com/doc/tutorial/pointers/
  // about parameter of CV_LOAD_IMAGE_COLOR
  //
  //http://opencv.jp/opencv-2.1/cpp/reading_and_writing_images_and_video.html#cv-imread
  // CV_LOAD_IMAGE_COLOR = 1, which make imread to read color image
  // if you change it to , then it will read grayscale image
  //src_img = cv::imread("./pre.png", CV_LOAD_IMAGE_COLOR);
  cv::Mat src_img = cv::imread("./warota.jpg", 1);
  
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
  {
    print("cannot open correctly");
    return -1;
  }else{
    print("usb cammera connected");
  }
  cv::Mat frame;
  cv::namedWindow("capture", CV_WINDOW_AUTOSIZE);
  while(cap.isOpened()){
    cap.read(frame);
    auto frame_overridden = face_detector(frame);
    cv::imshow("Capture", frame_overridden);
    cv::waitKey(5);
  }
}

cv::Mat convert2gray(const cv::Mat& src_img)
{
  cv::Mat img_;
  cv::cvtColor(src_img, img_, CV_BGR2GRAY);
  return img_;
}


cv::Mat gray_binarize(const cv::Mat& src_img)
{
  cv::Mat img_, img__;
  cv::cvtColor(src_img, img_, CV_BGR2GRAY);
  cv::threshold(img_, img__, 90, 255, CV_THRESH_BINARY);
  // all pixel grayscale val of which exceeds 90 will be set to 255;
  return img__;
}

// if src_img is not gray, the process will be done for each channesl
std::vector<cv::Mat> three_functions(const cv::Mat& src_img)
{
  int bit_depth = CV_32F; // TODO change it!!
  std::vector<cv::Mat> vec_img(3);
  cv::Mat tmp_img_, tmp_img;
  //http://opencv.jp/opencv-2svn/cpp/image_filtering.html?highlight=sobel#cv-sobel
  // here CV_32F is bit depth, 
  cv::Sobel(src_img, tmp_img_, CV_32F, 1, 1);
  cv::convertScaleAbs(tmp_img_, tmp_img, 1, 0);
  vec_img[0] = tmp_img;

  cv::Laplacian(src_img, tmp_img_, CV_32F, 3);
  cv::convertScaleAbs(tmp_img_, tmp_img, 1, 0);
  vec_img[1] = tmp_img;

  cv::Canny(src_img, tmp_img, 50.0, 200.0);
  vec_img[2] = tmp_img;
  return vec_img;
}

cv::Mat face_detector(const cv::Mat& src_img)
{
  const std::string cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml";
  cv::CascadeClassifier cascade;
  static std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 255), cv::Scalar(0, 128, 255),
    cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0),
    cv::Scalar(255, 128, 0), cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255)
  };

  static cv::Scalar colors_[] = {
    {0, 0, 255}, {0, 128, 255},
    {0, 255, 255}, {0, 255, 0},
    {255, 128, 0}, {255, 255, 0},
    {255, 0, 0}, {255, 0, 255}
  };

  std::vector<cv::Rect> faces;
  cascade.load(cascade_name);
  
  auto src_gray = convert2gray(src_img);
  cv::equalizeHist(src_gray, src_gray);
  
  // where | referes pipe operator
  cascade.detectMultiScale(src_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(40, 40));
  cv::Mat src_gray_overriden = src_img.clone(); 
  for(std::vector<cv::Rect>::iterator r = faces.begin(); r!=faces.end(); r++){
    cv::Point center;
    int radius;
    center.x = cvRound(r->x + r->width*0.5);
    center.y = cvRound(r->y + r->height*0.5);
    radius = cvRound((r->width + r->height)*0.25);
    int i = std::distance(faces.begin(), r);//TODO 
    print(i);
    cv::circle(src_gray_overriden, center, radius, colors[0], 3, 8, 0);
  }
  return src_gray_overriden;
}

