#include <MiniDNN.h>        //Neural Network Lib
#include "matplotlibcpp.h"  //plot lib
#include <fstream>          //file RW
#include <iostream>         
#include <algorithm>        //shuffle
#include <random>           //seed generation
#include <vector>           //std::vector

//using namespace MiniDNN;
namespace plt = matplotlibcpp;

typedef Eigen::MatrixXd Matrix;
//typedef Eigen::VectorXd Vector;

//read from csv and return a eigen Matrix
template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    unsigned int rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}


int main()
{
    
    //create input Matrix from file:
    Matrix inputMx = load_csv<Matrix>("input.csv");
    std::cout << "Rows: " << inputMx.rows() << " Cols: " << inputMx.cols() << '\n';

    std::cout << "First 2 rows of Input Matrix = \n";
    for (int i(0); i < 2; ++i) {
        for (int j(0); j < inputMx.cols(); ++j) {
            std::cout << inputMx(i, j) << ' ';
        }
        std::cout << '\n';
    }

    //input vector for plotting at the end
    std::vector<double> in;
    in.reserve(inputMx.rows());
    
    for (int i(0); i < inputMx.rows(); ++i) {
        in.push_back(i);
    }

    std::cout << "Inputvector: \n";
    std::cout << "Size: " << in.size() << " Capacity: " << in.capacity() << '\n';
/*    for (std::vector<double>::const_iterator i = in.begin(); i != in.end(); ++i) {
        std::cout << *i << ' ';
    }*/

    inputMx.transposeInPlace();
    std::cout << "First 2 columns of Transposed Input Matrix: \n";
    for (int i(0); i < inputMx.rows(); ++i) {
        for (int j(0); j < 2; ++j) {
            std::cout << inputMx(i, j) << ' ';
        }
        std::cout << '\n';
    }    
    
    std::cout << "Rows: " << inputMx.rows() << " Cols: " << inputMx.cols() << '\n';

    //normalize inputdata
    //for(int i(0); i < inputMx.rows(); ++i) {
    //    inputMx.row(i).normalize();
    //}
    std::cout << "First 2 columns of the Normalized Input Matrix: \n";
    for (int i(0); i < inputMx.rows(); ++i) {
        for (int j(0); j < 2; ++j) {
            std::cout << inputMx(i, j) << ' ';
        }
        std::cout << '\n';
    }   
    
    //create Matrix for prediction
    Matrix predinput = inputMx;
/*
    //shuffle the data:
    //create random seeds
    std::random_device r;
    std::seed_seq rng_seed{r(), r(), r(), r(), r(), r(), r(), r()};
    //create random engines with the rng seed
    std::mt19937 eng1(rng_seed);
    auto eng2 = eng1;
    //create permutation Matrix
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(inputMx.cols());
    permX.setIdentity();
    std::shuffle(permX.indices().data(), permX.indices().data()+permX.indices().size(), eng1);
    inputMx = inputMx * permX;
//    std::cout << "shuffled columns Matrix: \n";
//    std::cout << inputX << '\n';
*/
    //create output Matrix from file
    Matrix outputMx = load_csv<Matrix>("output.csv");
    std::cout << "Rows: " << outputMx.rows() << " Cols: " << outputMx.cols() << '\n';

    std::cout << "First 2 rows of Output Matrix = \n";
    for (int i(0); i < 2; ++i) {
        for (int j(0); j < outputMx.cols(); ++j) {
            std::cout << outputMx(i, j) << ' ';
        }
        std::cout << '\n';
    }

    outputMx.transposeInPlace();
    std::cout << "First 2 columns of Transposed Output Matrix: \n";
    for (int i(0); i < outputMx.rows(); ++i) {
        for (int j(0); j < 2; ++j) {
            std::cout << outputMx(i, j) << ' ';
        }
        std::cout << '\n';
    }    
    
    //normalize outputdata
    //for(int i(0); i< outputMx.rows(); ++i) { //in this case only one row => output n=1
    //    outputMx.row(i).normalize();
    //}
//    std::cout << "Normalized Output Matrix: \n";
//    std::cout << outputY << '\n';

    //create output vector for plotting at the end
    std::vector<double> out;
    out.reserve(outputMx.cols());
    for (int i(0); i < outputMx.cols(); ++i) {
        out.push_back(outputMx(0, i));
    }

    std::cout << "Rows: " << outputMx.rows() << " Cols: " << outputMx.cols() << '\n';

    //create permutation Matrix
   /* Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permY(outputMx.cols());
    permY.setIdentity();
    std::shuffle(permY.indices().data(), permY.indices().data()+permY.indices().size(), eng2);
    outputMx = outputMx * permY; */
//    std::cout << "shuffled columns Matrix: \n";
//    std::cout << outputY << '\n';
    
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
    MiniDNN::Network net;
    //Create three layers
    //Layer 1 -- fully connected, input = input size of Matrix
    MiniDNN::Layer* layer1 = new MiniDNN::FullyConnected<MiniDNN::Tanh>(4, 8);
    MiniDNN::Layer* layer2 = new MiniDNN::FullyConnected<MiniDNN::Tanh>(8, 2);
    MiniDNN::Layer* layer3 = new MiniDNN::FullyConnected<MiniDNN::Identity>(2, 1);

    //Add layers to the network
    net.add_layer(layer1);
    net.add_layer(layer2);
    net.add_layer(layer3);

    //set output layer
    net.set_output(new MiniDNN::RegressionMSE());
    //Create optimizer object
    MiniDNN::RMSProp opt;
    opt.m_lrate = 0.001;
    //set callback function object (output learning metrics)
    MiniDNN::VerboseCallback callback;
    net.set_callback(callback);
    //Initialize parameters with N(0, 0.01²) using random seed 87892136
    net.init(0, 0.01, 87892136);
    //Fit the model with a batch size of 10, running 1000 epochs with random seed 123456
    net.fit(opt, inputMx, outputMx, 30, 100, 123456);
     
    //Save Model to File for importing later
    //net.export_net("./NetFolder/", "NetFile");
    //Create new Network with file
//    Network netFromFile;
    //Read structure and parameters from file
//    netFromFile.read_net("./NetFolder/", "NetFile");
    //Obtain prediction -- each column is an observation
//    std::cout << net.predict(inputX) << '\n';
//    std::cout << netFromFile.predict(inputX) - net.predict(inputX) << '\n';
    //
/*
    //Obtain prediction -- each column is an observation
    std::cout << "Input: \n" << inputX << '\n';
    Matrix pred = net.predict(inputX);
    std::cout << "Prediction: \n" << net.predict(inputX) << '\n';
*/    
    // Layer objects will be freed by the network object,
    // so do not manually delete them

    //Compare prediction and real data and plot it
    std::cout << "Predinput rows: " << predinput.rows() << " Cols: " << predinput.cols() << '\n';
    Matrix pred = net.predict(predinput);
    std::cout << "Prediction Matrix Size:\n";
    std::cout << "Rows: " << pred.rows() << " Cols: " << pred.cols() << '\n';
    std::cout << "First 3 values of pred: ";
    for (int i(0); i < 3; ++i) {
        std::cout << pred(0, i) << ' ';
    }

    //create prediction vector
    std::vector<double> predvector;
    predvector.reserve(pred.cols());
    for (int i(0); i < pred.cols(); ++i) {
        predvector.push_back(pred(0, i));
    }



    std::cout << "Outputvector: \n";
    std::cout << "Size: " << out.size() << " Capacity: " << out.capacity() << '\n';
    plt::figure_size(1200, 780);
    plt::named_plot("Leistung(orig.) [W]", in, out);
    plt::named_plot("Leistung(pred.) [W]", in, predvector);
    plt::title("Leistungsdaten vor Training");
    plt::legend();
    plt::show();

    return 0;
}