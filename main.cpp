#include <MiniDNN.h>        //Neural Network Lib
#include <fstream>          //file RW
#include <iostream>         
#include <algorithm>        //shuffle
#include <random>           //seed generation
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
    /*
    //normalize data
    for(int i(0); i < inputX.cols(); ++i) {
        inputX.col(i).normalize();
    }
    std::cout << "Normalized Output Matrix: \n";
    std::cout << inputX << '\n';
*/
    //shuffle the data:
    //create random seeds
    std::random_device r;
    std::seed_seq rng_seed{r(), r(), r(), r(), r(), r(), r(), r()};
    //create random engines with the rng seed
    std::mt19937 eng1(rng_seed);
    auto eng2 = eng1;
    //create permutation Matrix
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(inputX.cols());
    permX.setIdentity();
    std::shuffle(permX.indices().data(), permX.indices().data()+permX.indices().size(), eng1);
    inputX = inputX * permX;
    std::cout << "shuffled columns Matrix: \n";
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

    //create permutation Matrix
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permY(outputY.cols());
    permY.setIdentity();
    std::shuffle(permY.indices().data(), permY.indices().data()+permY.indices().size(), eng2);
    outputY = outputY * permY;
    std::cout << "shuffled columns Matrix: \n";
    std::cout << outputY << '\n';
    /*
    //normalize data
    for(int i(0); i< outputY.cols(); ++i) {
        outputY.col(i).normalize();
    }
    std::cout << "Normalized Output Matrix: \n";
    std::cout << outputY << '\n';
    */
   /* //test normalize with initialized matrix
    Eigen::Matrix<double, 7, 3> A = (Eigen::Matrix<double, 7, 3>() << 
    1, 2, 3,
    3, 4, 5,
    4, 5, 6,
    1, 2, 3,
    10, 11, 12,
    20, 20, 20,
    3, 4, 5
    ).finished();
    A.col(0).normalize();
    std::cout << "First column normalized: \n";
    std::cout << A << '\n';
    */
 
    //Construct a network object
    Network net;
    //Create three layers
    //Layer 1 -- fully connected, input = input size of Matrix
    Layer* layer1 = new FullyConnected<Identity>(inputX.rows(), 20);
    Layer* layer2 = new FullyConnected<Tanh>(20, 10);
    Layer* layer3 = new FullyConnected<Tanh>(10, outputY.rows());

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
    //Initialize parameters with N(0, 0.01²) using random seed 87892136
    net.init(0, 0.01, 87892136);
    //Fit the model with a batch size of 5, running 8000 epochs with random seed 123456
    net.fit(opt, inputX, outputY, 5, 8000, 123456);
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
    std::cout << "Input: \n" << inputX << '\n';
    Matrix pred = net.predict(inputX);
    std::cout << "Prediction: \n" << net.predict(inputX) << '\n';
    
    // Layer objects will be freed by the network object,
    // so do not manually delete them

    return 0;
}