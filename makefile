all: ann

ann: main.o InputLayer.o Layer.o MultiLayerPerceptron.o OutputLayer.o
	g++ main.o InputLayer.o Layer.o MultiLayerPerceptron.o OutputLayer.o -o ann

main.o: main.cpp
	g++ -c main.cpp

InputLayer.o: InputLayer.cpp
	g++ -c InputLayer.cpp

Layer.o: Layer.cpp
	g++ -c Layer.cpp

MultiLayerPerceptron.o: MultiLayerPerceptron.cpp
	g++ -c MultiLayerPerceptron.cpp

OutputLayer.o: OutputLayer.cpp
	g++ -c OutputLayer.cpp

clean:
	del *.o ann.exe

run:
	ann.exe
