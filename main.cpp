/*
	This file is part of NNF2.

	NNF2 is free software: you can redistribute it and/or modify it
	under the terms of the GNU General Public License as published 
	by the Free Software Foundation, either version 3 of the License,
	or (at your option) any later version.

	NNF2 is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with NNF2.  If not, see <http://www.gnu.org/licenses/>.
*/

// example: XOR function
// this simple application shows how to create layers of neurons, connect them
// and build a neural network to perform some task

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include "MultiLayerPerceptron.h"
#include "Sigmoid.h"
#include "Heaviside.h"

using namespace std;
using namespace neural;

#define XOR(a, b) (((a) && !(b)) || (!(a) && (b)))

int main()
{
	// seed random number generator
	srand(time(NULL));
	
	// transfer functions
	Sigmoid sigmoid(2.0f); // sigmoid of parameter 2.0
	Heaviside heaviside; // heaviside of default parameter 0
	
	// learning rate and initial weight range
	float eps = 0.5f;
	float range = 1.0f;
	
	// input layer has 2 neurons, uses sigmoid transfer function
	InputLayer il(2, heaviside);
	
	// hidden layer is connected to input layer, has 4 neurons, uses sigmoid, 
	// has learning rate 'eps' and initial weights in (-range, range)
	Layer hl(&il, 2, sigmoid, eps, range);
	
	// output layer is connected to hidden layer, has 1 neuron, uses heaviside
	// transfer function
	OutputLayer ol(&hl, 1, heaviside, eps, range);
	
	// MLP network constructor takes input and output layers and 
	// a NULL-terminated list of hidden layers in the same order 
	// they were connected
	MultiLayerPerceptron mlp(&il, &ol, &hl, NULL);
	
	// a simple way to train the network is to generate some examples
	// of input-output couples
	for (int epochs = 0; epochs < 100; ++epochs)
		for (int i = 0; i <= 1; ++i)
			for (int j = 0; j <= 1; ++j) {
				float input[2] = {i, j};
				float desired_output = XOR(i, j);
				mlp.train(input, &desired_output);	
			}
	
	// then we can test the network fitness so far
	float input[2] = {1, 0};
	float output;
	mlp.compute(input, &output);
	cout << "XOR(1, 0) = " << output << endl;
	
	// another way to achieve training involves creating a file of this form:
	// 2 1		<-- number of input and output neurons, respectively
	// 4		<-- number of test cases in the file
	// 0 0		<-- input values of test case 1
	// 0		<-- output values of test case 1
	// 0 1
	// 1
	// 1 0
	// 1 
	// 1 1
	// 0
	ifstream datafile("xor.txt");
	
	// trains the network and returns true if it reached a mean square error
	// under 0.001f in no more than 100000 epochs
	bool success = mlp.optimize(datafile, 0.001f, 100000);
	cout << "Success: " << success << endl;

	// save the network on a text file
	ofstream savefile("net.txt");
	mlp.save(savefile);
	savefile.close();
	
	// load the network from a text file
	ifstream loadfile("net.txt");
	mlp.load(loadfile);
	loadfile.close();
	
	// let's see if the network has actually learned the XOR function
	for (int i = 0; i <= 1; ++i)
		for (int j = 0; j <= 1; ++j) {
			input[0] = i;
			input[1] = j;
			mlp.compute(input, &output);
			cout << "XOR(" << i << ", " << j << " )= " << output << endl;;
		}
	
	return 0;
}
