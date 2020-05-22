#include <MiniDNN.h>
#include <fstream>
#include <iostream>
using namespace MiniDNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;



int main()
{
    
    //Count Cols and rows of input:
    std::ifstream fin("input.csv");
    int val = 0, rows = 0, cols = 0, numItems = 0;

    while(fin.peek() != '\n' && fin >> val) {
        std::cout << val << ' ';
        ++numItems;
    }
    cols = numItems;

    std::cout << '\n';
    while(fin >> val) {
        ++numItems;
        std::cout << val << ' ';
        if(numItems % cols == 0) std::cout << '\n';
    }

    if(numItems > 0) {
        rows = numItems/cols;
        std::cout << "rows = " << rows << ", cols = " << cols << '\n';
    }
    else {
        std::cout << "data reading failed\n";
    }
    fin.close();
    fin.clear();

    //create input Matrix from file:
    Matrix inputX = Matrix::Zero(rows, cols);
    fin.open("input.csv");
    if(fin.is_open()) {
        for(int row = 0; row < rows; ++row) {
            for(int col = 0; col < cols; ++col) {
                double item = 0.0;
                fin >> item;
                inputX(row, col) = item;
            }
        }
        fin.close();
        fin.clear();
    }
    std::cout << "Input Matrix = \n";
    std::cout << inputX << '\n';
    inputX.transposeInPlace();
    std::cout << "Transposed Input Matrix: \n";
    std::cout << inputX << '\n';

    //Count Cols and rows of output:
    val = 0, rows = 0, cols = 0, numItems = 0;
    fin.open("output.csv");
    while(fin.peek() != '\n' && fin >> val) {
        std::cout << val << ' ';
        ++numItems;
    }
    cols = numItems;
    std::cout << '\n';
    while(fin >> val) {
        ++numItems;
        std::cout << val << ' ';
        if(numItems % cols == 0) std::cout << '\n';
    }
    if(numItems > 0) {
        rows = numItems / cols;
        std::cout << "rows = " << rows << ", cols = " << cols << '\n';
    }
    else {
        std::cout << "output data reading failed\n";
    }
    fin.close();
    fin.clear();

    //create output Matrix from file:
    Matrix outputY = Matrix::Zero(rows, cols);
    fin.open("output.csv");
    if(fin.is_open()) {
        for(int row = 0; row < rows; ++row) {
            for(int col = 0; col < cols; ++col) {
                double item = 0.0;
                fin >> item;
                outputY(row, col) = item;
            }
        }
        fin.close();
    }
    std::cout << "Output Matrix = \n";
    std::cout << outputY << '\n';
    outputY.transposeInPlace();
    std::cout << "Transposed Output Matrix = \n";
    std::cout << outputY << '\n';

    //Construct a network object
    Network net;
    //Create three layers
    //Layer 1 -- fully connected, input = input size of Matrix
    Layer* layer1 = new FullyConnected<Identity>(10, 20);
    Layer* layer2 = new FullyConnected<Tanh>(20, 10);
    Layer* layer3 = new FullyConnected<Tanh>(10, 1);

    //Add layers to the network
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);

    //set output layer
    net.set_output(new RegressionMSE());
    //Create optimizer object
    RMSProp opt;
    opt.m_lrate = 0.001;
    //set callback function object (output learning metrics)
    VerboseCallback callback;
    net.set_callback(callback);
    //Initialize parameters with N(0, 0.01²) using random seed 872164782164892136
    net.init(0, 0.01, 872164782164892136);
    //Fit the model with a batch size of 100, running 10 epochs with random seed 123
    net.fit(opt, inputX, outputY, 100, 10, 123);
/*     
    //Save Model to File for importing later
    net.export_net("Netfolder", "NetFile");
    //Create new Network with file
    Network netFromFile;
    //Read structure and parameters from file
    netFromFile.read_net("./NetFolder/", "NetFile");
    //Obtain prediction -- each column is an observation
    std::cout << net.predict(inputX) << '\n';
    std::cout << netFromFile.predict(inputX) - net.predict(inputX) << '\n';
     */

    //Obtain prediction -- each column is an observation
    Matrix pred = net.predict(inputX);
    std::cout << net.predict(inputX) << '\n';
    
    // Layer objects will be freed by the network object,
    // so do not manually delete them

    return 0;
}