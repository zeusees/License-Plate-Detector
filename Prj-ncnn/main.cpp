#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "LPDetector.h"
#include <dirent.h>
using namespace std;

void readFileNameInDir(string strDir, vector<string>& vFileFullPath)
{
       struct dirent* pDirent;
       DIR* pDir = opendir(strDir.c_str());
       if (pDir != NULL)
       {
              while ((pDirent = readdir(pDir)) != NULL)
              {
                     string strFileName = pDirent->d_name;
                     string strFileFullPath = strDir + "/" + strFileName;
                     vFileFullPath.push_back(strFileFullPath);
              }
              vFileFullPath.erase(vFileFullPath.begin(), vFileFullPath.begin() +  2);    //前两个存储的是当前路径和上一级路径，所以要删除
       }
}


int main(int argc, char** argv)
{


    vector<string> files; 
    char * filePath = "/home/zeusee/projects/testimg/";
 
    readFileNameInDir(filePath,files);
   
    int size = files.size();  
    for (int i = 0;i < size;i++)  
    {  
        cout<<files[i].c_str()<<endl;  
    }

    
    string param = "../model/mnet_plate.param";
    string bin = "../model/mnet_plate.bin";
    const int max_side = 640;

    // retinaface
    Detector detector(param, bin, true);
    Timer timer;
    int cnt = 0;
    for (int i = 0;i < size;i++)  
    {  

        cv::Mat img = cv::imread(files[i]);

        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = 1.0 ;//max_side/long_side;
        cv::Mat img_scale;
        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));


        std::vector<bbox> boxes;

        timer.tic();

        detector.Detect(img_scale, boxes);
        timer.toc("----total timer:");

        // draw image
        
        for (int j = 0; j < boxes.size(); ++j) {

            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);

            cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
        }
        
        cv::Mat dst;
        cv::resize(img, dst, cv::Size(1280, 720), cv::INTER_NEAREST);
        cv::imshow("img",dst);
        cv::waitKey(0);
    }
    return 0;
}

