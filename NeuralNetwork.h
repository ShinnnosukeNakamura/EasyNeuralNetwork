#pragma once
using namespace std;
class NeuralNetwork
{
private:
	double **output_value;
	double ***weight_forward;
	double **bias_forward;
	double ***weight_back;
	double **bias_back;
	double **relative_error;
	double eqsiron;
	double eta;
	double alpha;
	int N;
	int number_learn;
	int num_input;
	int num_hidden;
	int num_out;
	int gain;
	int *number_hidden_layer;
	int layer_size;
public:
	NeuralNetwork(int,int,int,int*,double,double,double,int,int);
	~NeuralNetwork();
	void train(double**, double**);
	void predict(double**,double**);
	void forwardPropagation(double*);
	void backPropagation(double*);
	void initialize();
	double sigmoid(double);

};

