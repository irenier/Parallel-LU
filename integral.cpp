#include <array>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <openmpi/mpi.h>
#include <vector>

using namespace Eigen;
using namespace std;

decltype(auto) x_exact(const auto &t) { return exp(t); }
decltype(auto) y(const auto &t) { return (exp(t + 1.) - 1.) / (t + 1.); }

int main(int argc, char **argv) {
  setNbThreads(8);

  const int N = 1000;        // 区间数量
  const double h = 1. / N;   // 步长
  const double alpha = 1e-3; // 正则化参数

  // 组装矩阵 K
  MatrixXd K(N + 1, N + 1);
  for (int i = 0; i < N + 1; i++) {
    K(i, 0) = 0.5;
    K(i, N) = 0.5 * exp(i * N * pow(h, 2));
    for (int j = 1; j < N; j++) {
      K(i, j) = exp(i * j * pow(h, 2));
    }
  }
  K = K.array() * h;

  // 组装右端项 b
  VectorXd b(N + 1);
  for (int i = 0; i < N + 1; i++) {
    b(i) = y(i * h);
  }

  // 若使用 Eigen 库运算
  // 计算 K^T * K
  MatrixXd KtK = K.transpose() * K;

  // 计算 alpha * I + K^T * K
  MatrixXd alphaI = MatrixXd::Identity(N + 1, N + 1) * alpha;
  MatrixXd K_regular = alphaI + KtK;

  // 计算 K^T * y
  VectorXd b_regular = K.transpose() * b;

  // 进行求解
  VectorXd x = K_regular.lu().solve(b_regular);

  // 定义用于误差计算的精确解
  VectorXd x_e(N + 1);
  for (int i = 0; i < N + 1; i++) {
    x_e(i) = x_exact(i * h);
  }

  // 将解写入文件
  std::ofstream file("../sol.txt");
  if (file.is_open()) {
    file << x << std::endl;
    file.close();
  } else {
    std::cerr << "无法打开文件！" << std::endl;
  }

  // 计算并打印相对误差
  double error = (x - x_e)(seq(1, last - 1)).norm() / x_e.norm();
  cout << "相对误差: " << error << endl;

  return 0;
}
