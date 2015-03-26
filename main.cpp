#include<stdio.h>
#include<iostream>
#include"NeuralNetwork.h"
using namespace std;

int main(){
	cout << "Start\n";
	//データセット
	double x[6][3] = {	{0,0,0},
						{1,0,1},
						{1,1,1},
						{1,1,0},
						{1,0,0},
						{0,0,1} };
	double y[6][1] = { { 0 }, { 0 }, { 1 }, { 0 }, { 1 }, {1} };

	double **xx, **yy,**out;
	xx = new double*[6];
	yy = new double*[6];
	for (int j = 0; j < 6; j++){
		xx[j] = new double[3];
		yy[j] = new double[1];
	}
	out = yy;
	for (int j = 0; j < 6;j++){
		yy[j][0] = y[j][0];
		for (int k = 0; k < 3; k++){
			xx[j][k] = x[j][k];
		}
	}

	int *hid = new int[2]{5,2};

	NeuralNetwork neuralnet(3, 2, 1, hid, 1.0, 0.01, 0.01,6, 800000);
	neuralnet.train(xx, yy);
	neuralnet.predict(xx, out);
	for (int j = 0; j < 6; j++){
		cout << "Value  ; " << out[j][0] << endl;
	}
	/*
	実行結果 win8 pro 64bit AMD A8 3.0GHz 8.00GB 
	Visual Studio 2013では Releaseモードでないと学習が上手くいかない
		value : 0.0243593
		value : 0.301272
		value : 0.980685
		value : 0.0256001
		value : 0.9763
		value : 0.976293
	*/
	getchar();
	return 0;
}