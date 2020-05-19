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

    
    return 0;
}