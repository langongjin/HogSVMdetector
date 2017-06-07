#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <sys/time.h>

using namespace cv;
using namespace std;

Mat img;

HOGDescriptor initialize_descriptor_by_file(ifstream &fin){

    HOGDescriptor myHOG(Size(112,24),Size(16,16),Size(16,8),Size(8,8),9);//HOG检测器，设置HOGDescriptor的检测子,用来计算HOG描述子的

    float val = 0.0f;
    vector<float> myDetector;
    while(!fin.eof())
    {
        fin>>val;
        myDetector.push_back(val);
    }
    fin.close();
    myDetector.pop_back();

    myHOG.setSVMDetector(myDetector);

    return myHOG;
}

Mat HogDetectMulti(Mat &src,HOGDescriptor myHOG){
    Mat drawed_img;
    src.copyTo(drawed_img);

    vector<Rect> found;// vector array of foundlocation

    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(0,0), 1.05, 2); //1.05->36 levels, 1.1->17 levels, 1.12->14 levels

    //go through all of the detected targets, get the hard examples
    int len = found.size();
    for(int i=0; i < found.size(); i++)
    {
        //resize the window inside the image, because sometime the window out of the image
        Rect r = found[i];

//        if(r.x < 0)
//            r.x = 0;
//        if(r.y < 0)
//            r.y = 0;
//        if(r.x + r.width > src.cols)
//            r.width = src.cols - r.x;
//        if(r.y + r.height > src.rows)
//            r.height = src.rows - r.y;

        if (len == 1)
            rectangle(drawed_img, r.tl(), r.br(), Scalar(0,255,255), 1);
        else
        {
            float area_inter, area_r=(float)r.area();
            for( int j = 0; j < len; j++)
            {
                area_inter = (float) (Rect(r & found[j]).area());
                if(j > i && area_inter > area_r*0.5) //j!=i, exception itself
                {
                    Point mintl, maxbr;
                    mintl.x = min(found[i].tl().x,found[j].tl().x);
                    mintl.y = min(found[i].tl().y,found[j].tl().y);
                    maxbr.x = max(found[i].br().x,found[j].br().x);
                    maxbr.y = max(found[i].br().y,found[j].br().y);
                    rectangle(drawed_img, mintl, maxbr, Scalar(0,255,255), 1);
                }
            }
        }
    }
    return drawed_img;
}

int main()
{
    ifstream fin("/Users/lan/Desktop/TarReg/svm/svmrobot/training/HOGDetector0502robot.txt", ios::in);
    HOGDescriptor myHOG=initialize_descriptor_by_file(fin);

    struct timeval timeStart, timeEnd;
    double timeDiff;

    VideoCapture cap(0); //open the first camera

    if(!cap.isOpened()) //check for successful open, or not
    {
        cerr << "Can not open a camera or file." << endl;
        return -1;
    }
    int rate = cap.get(CV_CAP_PROP_FPS); //获取帧率

    cout << "frame of camera : " << rate << endl;

    bool stop = false; //定义一个用来控制读取视频循环结束的变量
    while(!stop)
    {
        cap >> img; //read a frame image and save to the Mat img

        gettimeofday(&timeStart,NULL);
        imshow("Detected",HogDetectMulti(img,myHOG));
        gettimeofday(&timeEnd,NULL);
        timeDiff = 1000*(timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_usec - timeStart.tv_usec)/1000; //tv_sec: value of second, tv_usec: value of microsecond
        cout << "Time for one frame : " << timeDiff << " ms" << endl;

        int c = waitKey(10); //waiting for 5 milliseconds to detect the pression, if waitKey(0) meaning always waiting until any pression
        if((char) c == 27) // press escape to exit the imshow
        {
            stop = true;
        }
    }
    //waitKey(); //注意：imshow之后必须加waitKey，否则无法显示图像
    return 0;
}