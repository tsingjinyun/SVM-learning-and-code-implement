#include <cv.h>
#include <highgui.h>
#include "ml.h"
#include <stdio.h>
#include"iostream.h"
float a[2500];
int main(int argc, char **argv)
{
int i, j, ii, jj;
int width= 50, height= 50;      /*样本图像的尺寸大小*/
int image_dim= width* height;
int pimage_num= 41;         /*正样本数*/ 
int nimage_num= 61;        /*负样本数*/ 
int all_image_num= pimage_num+ nimage_num;
IplImage *img_org;
IplImage *sample_img;
//int res[all_image_num];
int res[102];
//float data[all_image_num* image_dim];
float data[255000];
CvMat data_mat, res_mat;
CvTermCriteria criteria;
CvSVM svm= CvSVM ();
CvSVMParams param;
char filename[65];







// (1) 读取正样本数据
for (i= 0; i< pimage_num; i++) {
    sprintf(filename, "灰度样本\\正样本\\1 (%03d).jpg",i);     //产生正样本的文件路径
   printf(filename);
    img_org= cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
   if(img_org!=NULL)
      cout<<"成功"<<endl;
   else
      cout<<"失败"<<endl;
    sample_img= cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    cvResize(img_org, sample_img);
    cvSmooth(sample_img, sample_img, CV_GAUSSIAN, 3, 0, 0, 0);
   for (ii= 0; ii< height; ii++) {
     for (jj= 0; jj< width; jj++) {
        data[i* image_dim+ (ii* width) + jj] =
         float ((int) ((uchar) (sample_img->imageData[ii* sample_img->widthStep+ jj])) / 255.0);
     }
   }
    res[i] = 1;
}
    //读取正样本的图像群，将各像素值转化成float数据类型。为了方便，预先在"positive/目录下准备了正样本图像，图像名用3位连续的数字名标识（000,001,002，·····）。首先，将读取的各图像转换成同一尺寸大小（28×30），为了减轻噪声的影响，对图像作了平滑化处理。然后，为了利用各像素亮度值(这里的图像作为等级图像被读取)的特征向量，将它变换成了数组。总之，对于一张图像的特征向量(图像宽度X图像长度)，准备了和样本图像张数相同的数量。"1"表示利用此特征向量的判别数值。此外还使用500张脸部图像的正样本(基本上是正面脸部图像，没有侧面的)。
    //在OpenCV里实装了利用haar-like特征的物体检测算法，由于利用它检测脸部的精度和处理速度都很不错，虽然脸部图像检测没有太大意义，但从获取样本的难易程度和理解程度考虑，此次利用脸部图像进行学习。

// (2) 读取负样本数据
  j= i;
for (i= j; i< j+ nimage_num; i++) {
    sprintf(filename, "灰度样本\\负样本\\1 (%03d).jpg", i-j);
    img_org= cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
    sample_img= cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    cvResize(img_org, sample_img);
    cvSmooth(sample_img, sample_img, CV_GAUSSIAN, 3, 0, 0, 0);
   for (ii= 0; ii< height; ii++) {
     for (jj= 0; jj< width; jj++) {
        data[i* image_dim+ (ii* width) + jj] =
         float ((int) ((uchar) (sample_img->imageData[ii* sample_img->widthStep+ jj])) / 255.0);
     }
   }
    res[i] = 0;
}
//读取负样本的图像群，和正样本一样，将它们转化成数组，并用"0"表示利用该特征向量的判别数值。另外使用1000张任意图像的负样本(全部是脸部以外的图像)。

// (3)SVM学习数据和参数的初始化
  cvInitMatHeader(&data_mat, all_image_num, image_dim, CV_32FC1, data);
  cvInitMatHeader(&res_mat, all_image_num, 1, CV_32SC1, res);
  criteria= cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
  param= CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);
//样本图像的像素值数组和判别值数组进行行列变换。为了样本学习对参数作了初始化处理。由于指定了非常适合的参数，也就有必要设定相应合适的参数。

// (4)SVM学习和数据保存
  svm.train(&data_mat, &res_mat, NULL, NULL, param);
  //svm.save("svm_image.xml");
  cvReleaseImage(&img_org);
  cvReleaseImage(&sample_img);



//利用正负样本的像素值和被指定的参数，根据svm.train()方式进行SVM学习。样本数:正样本500，负样本1000，特征向量: 28×30=840维。学习的SVM参数用根据svm.save()方法的XML形式的文件保存。如此页开始部分讲述的那样，为了使用保存和下载功能，有必要对OpenCV源代码作修改。
//第三个  根据图像各像素值转化成特征向量的SVM检测物体(脸部)
//读取用于学习的SVM参数，从相关图像中检测物体。

CvMat m;
//float a[2500];
float ret= -1.0;
float scale;
IplImage *src, *src_color, *src_tmp;
int sx, sy, tw, th;
int stepx= 3, stepy= 3;
double steps= 1.2;
int iterate;

// (1)图像的读取
src_color= cvLoadImage("待检测图像//11.jpg", CV_LOAD_IMAGE_COLOR);
src= cvLoadImage("待检测图像//11.jpg", CV_LOAD_IMAGE_GRAYSCALE);
if (src == 0||src_color==0) 
{
      cout<<"fail fail  fail"<<endl;
}
else
cout<<"OKOKOKOKOKOKOKOKOKOK"<<endl;
// (2)SVM数据的读取
  ///svm.load("svm_image.xml");
/*对读取图像的每部分进行处理 */
cvInitMatHeader(&m, 1, image_dim, CV_32FC1, NULL);
  tw= src->width;
  th= src->height;

for (iterate= 0; iterate< 1; iterate++) {
// (3) 缩小图像，并对当前图像部分作行列变换
    src_tmp= cvCreateImage(cvSize((int) (tw/ steps), (int) (th/ steps)), IPL_DEPTH_8U, 1);
    cvResize(src, src_tmp);
    tw= src_tmp->width;
    th= src_tmp->height;
   for (sy= 0; sy<= src_tmp->height- height; sy+= stepy) {
     for (sx= 0; sx<= src_tmp->width- width; sx+= stepx) {
       for (i= 0; i< height; i++) {
         for (j= 0; j< width; j++) {
            a[i* width+ j] =
             float ((int) ((uchar) (src_tmp->imageData[(i+ sy) * src_tmp->widthStep+ (j+ sx)])) / 255.0);
         cout<<j<<endl;
         }
       }
        cvSetData(&m, a, sizeof (float) * image_dim);
// (4)根据SVM的判定和结果绘图
        ret= svm.predict(&m);
      cout<<"ret is "<<ret<<endl;
       if ((int) ret== 1) {
          scale= (float) src->width/ tw;
          cvRectangle(src_color, cvPoint((int) (sx* scale), (int) (sy* scale)),
                       cvPoint((int) ((sx+ width) * scale), (int) ((sy+ height) * scale)), CV_RGB(255, 0, 0), 2);
       }
     }
   }
    cvReleaseImage(&src_tmp);
}

// (5)显示检测出的结果图像
  cvNamedWindow("svm_predict", CV_WINDOW_AUTOSIZE);
  cvShowImage("svm_predict", src_color);
  cvWaitKey(0);
  cvDestroyWindow("svm_predict");
  cvReleaseImage(&src);
  cvReleaseImage(&src_color);
  cvReleaseImage(&src_tmp);


return 0;
}