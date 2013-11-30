//https://github.com/glass5er/OpenCV-MachineLearning-SVM/blob/master/main.cpp

#include "StdAfx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

using std::vector;
using std::string;
using cv::Mat;

using std::cout;
using std::cerr;
using std::endl;

static bool verbose = false;

void split(string str, string delim, vector<string> &result)
{
  result.clear();
  int cut_pos;
  while( (cut_pos = str.find_first_of(delim)) != (int)string::npos ) {
    if(cut_pos > 0) {
      result.push_back(str.substr(0, cut_pos));
    }
    str = str.substr(cut_pos + 1);
  }
  if(str.length() > 0) {
    result.push_back(str);
  }
  return;
}

void
read_csv(const char *fname,
    std::vector<int> &flag,
    std::vector< std::vector<double> > &data)
{
  std::ifstream ifs;
  ifs.open(fname);
  //  no file -> abort  //
  if(!ifs.is_open()) {
    cerr << "no input file : " << fname << endl;
    return;
  }

  //  init data flags and segments  //
  data.clear();
  flag.clear();

  const int bufsize(8192);
  char buf[bufsize];
  for(int i=0; !ifs.eof(); i++) {
    vector<double> linedata(0);
    //  get training data flag (beginning of each line)   //
    ifs.getline(buf, sizeof(buf));
    vector<string> chunks;
    string str = string(buf);
    //  skip if no data exist  //
    if(str.size() <= 0) continue;
    split(str, string(","), chunks);
    //  get group of data  //
    string group = chunks[0];
    if( group.find("T") != string::npos ) {
      flag.push_back(1);
    }else if( group.find("F") != string::npos ) {
      flag.push_back(2);
    }else{
      flag.push_back(0);
    }
    //  @DEBUG : show training data flag  //
    if(verbose) {
      cout << i << " : " << flag[i] << " : " << buf << endl;
    }
    //  get data segments  //
    const int segments = (int)chunks.size() - 1;
    for(int j=0; j<segments; j++) {
      linedata.push_back(atof(chunks[j+1].c_str()));
      //  @DEBUG : show data of each column  //
      if(verbose) {
        cout << linedata[j] << " ";
      }
    }
    if(verbose) {
      cout << endl << endl;
    }
    //
    data.push_back(linedata);
  }
  ifs.close();
}

const char* keys =
{
  //  { short | long | init | note }  //
  "{1|||training dataset file name}"
  "{2|||prediction dataset file name}"
  "{c|classifier-num|10|number of weak classifier}"
  "{v|verbose|false|verbose mode}"
};

int main( int argc, const char** argv )
{
  //  @SETTING :   //
  static char printbuf[1024];
  setvbuf(stdout,printbuf,_IOLBF,sizeof(printbuf));

  //  parse options  //
  cv::CommandLineParser parser(argc, argv, keys);
  const string fname_train = parser.get<string>("1");
  const string fname_pred = parser.get<string>("2");
  verbose = parser.get<bool>("v");

  //  input file error -> exit  //
  bool run_flag = true;
  if(fname_train.empty()) {
    cerr << "no inputs for training." << endl;
    run_flag = false;
  }
  if(fname_pred.empty()) {
    cerr << "no inputs for prediction." << endl;
    run_flag = false;
  }
  if(!run_flag) {
    cerr << "Call :" << endl
         << "  ./sample_svm [training_file] [prediction_file]" << endl;
    return -1;
  }

  //  training data buffer  //
  std::vector< int > flagset;
  std::vector< std::vector<double> > dataset;
  read_csv(fname_train.c_str(), flagset, dataset);
  const int lines_tr = dataset.size();
  const int segments = dataset[0].size();

  //  dataset -> matrix  //
  cv::Mat flagmat(lines_tr, 1, CV_32SC1);
  cv::Mat datamat(lines_tr, segments, CV_32FC1);
  for(int i=0; i<lines_tr; i++) {
    flagmat.at<int>(i,0) = flagset[i];
    for(int j=0; j<segments; j++) {
      datamat.at<float>(i,j) = dataset[i][j];
    }
  }

  //  training (SVM)  //
  CvSVM svm;
  svm.train(datamat, flagmat);

  //  prediction data buffer  //
  std::vector< int > flag_tmp;
  std::vector< std::vector<double> > data_tmp;
  read_csv(fname_pred.c_str(), flag_tmp, data_tmp);
  const int samples = data_tmp.size();

  //  each sample -> predict //
  cv::Mat sample(1, segments, CV_32FC1);
  for(int i=0; i<samples; i++) {
    for(int j=0; j<segments; j++) {
      sample.at<float>(0,j) = data_tmp[i][j];
    }
    //  prediction (SVM)  //
    float res = svm.predict(sample);
    cout << "pred result[" << i << "] = " << res << endl;
  }

  return 0;
}