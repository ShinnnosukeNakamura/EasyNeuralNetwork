#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "NeuralNetwork.h"

#define Rlow             -0.30
#define Rhigh            0.30
#define urand()          ( (double)rand() / 0x7fff * (Rhigh - Rlow) + Rlow )

NeuralNetwork::NeuralNetwork(int n_in, int n_hid, int n_out,int *hid_l,double ga,double et,double al, int n_c, int l_n)
{
	gain = ga;
	eta = et;
	alpha = al;

	num_out = n_out;
	num_hidden = n_hid;
	num_input = n_in;
	N = n_c;
	number_learn = l_n;
	number_hidden_layer = hid_l;
	layer_size = n_hid;
	//o—Í‘w‚Ì•ªƒvƒ‰ƒX‚P‚µ‚Ä‚¢‚é
	layer_size += 1;
	

	output_value = new double*[layer_size];
	weight_forward = new double**[layer_size];
	weight_back = new double**[layer_size];
	bias_forward = new double*[layer_size];
	bias_back = new double*[layer_size];
	relative_error = new double*[layer_size];

	for (int x = 0; x < layer_size; x++){
		if (x == layer_size - 1){
			output_value[x] = new double[n_out];
			relative_error[x] = new double[n_out];
			bias_forward[x] = new double[n_out];
			bias_back[x] = new double[n_out];
			
		}
		else{
			output_value[x] = new double[number_hidden_layer[x]];
			relative_error[x] = new double[number_hidden_layer[x]];
			bias_forward[x] = new double[number_hidden_layer[x]];
			bias_back[x] = new double[number_hidden_layer[x]];
		}

	}

	//Œë·“`”À‚ÌŒvZ‚Å‚Íback‚Í‹t‚©‚çŒvZ‚µ‚È‚¯‚ê‚Î‚È‚ç‚È‚¢
	for (int x = 0; x < layer_size; x++){
		if (x == 0){
			weight_forward[x] = new double*[num_input];
			weight_back[x] = new double*[num_input];
			for (int y = 0; y < num_input; y++){
				weight_forward[x][y] = new double[number_hidden_layer[0]];
				weight_back[x][y] = new double[number_hidden_layer[0]];
			}
		}
		else{
			weight_forward[x] = new double*[number_hidden_layer[x - 1]];
			weight_back[x] = new double*[number_hidden_layer[x -1]];

			if (x != layer_size - 1){
				for (int y = 0; y < number_hidden_layer[x -1]; y++){
					weight_forward[x][y] = new double[number_hidden_layer[x]];
					weight_back[x][y] = new double[number_hidden_layer[x]];
				}
			}
			else{
				for (int y = 0; y < number_hidden_layer[x - 1]; y++){
					weight_forward[x][y] = new double[num_out];
					weight_back[x][y] = new double[num_out];
				}
			}
		}
		
	}
	
	initialize();
}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::initialize(){
	for (int x = 0; x < layer_size; x++){
		if (x == 0){
			for (int y = 0; y < num_input; y++){
				for (int z = 0; z < number_hidden_layer[0]; z++){
					weight_forward[x][y][z] = urand();
					weight_back[x][y][z] = 0;
					bias_forward[x][y] = urand();
					bias_back[x][y] = 0;
				}
			}
		}
		else{
			if (x != layer_size - 1){
				for (int y = 0; y < number_hidden_layer[x - 1]; y++){
					for (int z = 0; z < number_hidden_layer[x]; z++){
						weight_forward[x][y][z] = urand();
						weight_back[x][y][z] = 0;
						bias_forward[x][y] = urand();
						bias_back[x][y] = 0;
					}
				}
			}
			else{
				for (int y = 0; y < number_hidden_layer[x - 1]; y++){
					for (int z = 0; z < num_out; z++){
						weight_forward[x][y][z] = urand();
						weight_back[x][y][z] = 0;
						bias_forward[x][y] = urand();
						bias_back[x][y] = 0;
					}
				}
			}
		}

	}
}
void NeuralNetwork::train(double **x, double**y){
	for (int co = 0; co < number_learn; co++){
		for (int nu = 0; nu < N; nu++){
			forwardPropagation(x[nu]);
			backPropagation(y[nu]);
		}
	}
}
void NeuralNetwork::predict(double **x, double **y){
	for (int nu = 0; nu < N; nu++){
		forwardPropagation(x[nu]);
		for (int ou = 0; ou < num_out; ou++){
			y[nu][ou] = output_value[layer_size - 1][ou];
		}
	}
}

void NeuralNetwork::forwardPropagation(double *x){
	double sum;
	for (int layer = 0; layer < layer_size; layer++){
		if (layer == 0){
			for (int nel = 0;nel < number_hidden_layer[layer]; nel++){
				sum = 0;
				for (int el = 0; el < num_input; el++){
					sum += weight_forward[layer][el][nel] * x[el];
				}
				sum += bias_forward[layer][nel];
				output_value[layer][nel] = sigmoid(sum);
			}
		}
		else if(layer < layer_size -1){
			for (int nel = 0; nel < number_hidden_layer[layer]; nel++){
				sum = 0;
				for (int el = 0; el < number_hidden_layer[layer - 1]; el++){
					sum += weight_forward[layer][el][nel] * output_value[layer -1][el];

				}
				sum += bias_forward[layer][nel];
				output_value[layer][nel] = sigmoid(sum);
			}
		}
		else{
			for (int nel = 0; nel < num_out; nel++){
				sum = 0;
				for (int el = 0; el < number_hidden_layer[layer - 1]; el++){
					sum += weight_forward[layer][el][nel] * output_value[layer - 1][el];
				}
				sum += bias_forward[layer][nel];
				output_value[layer][nel] = sigmoid(sum);

			}
		}
		
	}
	
}

void NeuralNetwork::backPropagation(double *y){
	int l_c = 0;
	double sum = 0;
	for (int layer = 0; layer < layer_size; layer++){
		if (layer == 0){
			for (int el = 0; el < num_out; el++){
				relative_error[layer_size - 1][el] = (y[el] - output_value[layer_size - 1][el])*output_value[layer_size - 1][el] * (1 - output_value[layer_size - 1][el]);
			}
			for (int el = 0; el < number_hidden_layer[layer_size - 2]; el++){
				sum = 0;
				for (int bel = 0; bel < num_out; bel++){
					weight_back[layer_size - 1][el][bel] = eta * relative_error[layer_size - 1][bel] * output_value[layer_size - 2][el] + alpha * weight_back[layer_size - 1][el][bel];
					weight_forward[layer_size - 1][el][bel] += weight_back[layer_size - 1][el][bel];
					sum += relative_error[layer_size - 1][bel] * weight_forward[layer_size - 1][el][bel];
				}
				relative_error[layer_size - 2][el] = output_value[layer_size - 2][el] * (1 - output_value[layer_size - 2][el]) * sum;
			}
			for (int el = 0; el < num_out; el++){
				bias_back[layer_size - 1][el] = eta * relative_error[layer_size - 1][el] + alpha * relative_error[layer_size - 1][el];
				bias_forward[layer_size - 1][el] += bias_back[layer_size - 1][el];
			}
		}
		else if (layer < layer_size - 1){
			for (int el = 0; el < number_hidden_layer[layer_size - (layer + 2)]; el++){
				sum = 0;
				for (int bel = 0; bel < number_hidden_layer[layer_size - (layer + 1)]; bel++){
					weight_back[layer_size - (layer + 1)][el][bel] = eta * relative_error[layer_size - (layer + 1)][bel] * output_value[layer_size - (layer + 2)][el] + alpha * weight_back[layer_size - (layer + 1)][el][bel];
					weight_forward[layer_size - (layer + 1)][el][bel] += weight_back[layer_size - (layer + 1)][el][bel];
					sum += relative_error[layer_size - (layer + 1)][bel] * weight_forward[layer_size - (layer + 1)][el][bel];
				}
				relative_error[layer_size - (layer + 2)][el] = output_value[layer_size - (layer + 2)][el] * (1 - output_value[layer_size - (layer + 2)][el]) * sum;
			}
			for (int el = 0; el < number_hidden_layer[layer_size - (layer + 1)];el++){
				bias_back[layer_size - (layer + 1)][el] = eta * relative_error[layer_size - (layer + 1)][el] + alpha * relative_error[layer_size - (layer + 1)][el];
				bias_forward[layer_size - (layer + 1)][el] += bias_back[layer_size - (layer + 1)][el];
			}
		}
		else{
			for (int el = 0; el < num_input; el++){
				sum = 0;
				for (int bel = 0; bel < number_hidden_layer[0]; bel++){
					weight_back[0][el][bel] = eta * relative_error[0][el] + alpha * weight_back[0][el][bel];
					weight_forward[0][el][bel] += weight_back[0][el][bel];
				}
				for (int el = 0; el < number_hidden_layer[0]; el++){
					bias_back[0][el] = eta * relative_error[0][el] + alpha * relative_error[0][el];
					bias_forward[0][el] += bias_back[0][el];
				}
			}
		}
	}
}
double NeuralNetwork::sigmoid(double x){
	return 1 / (1 + exp(-gain*x));
}
