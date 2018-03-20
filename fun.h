#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include<stdlib.h>


using namespace cv;
using namespace std;

Mat *buildpyr(Mat im);
Mat reconpyr(Mat *im);