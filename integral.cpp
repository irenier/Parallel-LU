#include <array>
#include <cblas.h>
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

// 使用并行的 LU 分解求解线性系统 Ax = b
VectorXd parallel_lu_solve(const MatrixXd &A_global, const VectorXd &b_global,
                           MPI_Comm comm);

int main(int argc, char **argv) {
  setNbThreads(8);

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

  // 使用 CBLAS 计算 K^T * K
  MatrixXd KtK(N + 1, N + 1);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N + 1, N + 1, N + 1, 1.0,
              K.data(), N + 1, K.data(), N + 1, 0.0, KtK.data(), N + 1);

  // 构建正则化系统矩阵: alpha * I + K^T * K
  MatrixXd alphaI = MatrixXd::Identity(N + 1, N + 1) * alpha;
  MatrixXd K_regular = alphaI + KtK;

  // 构建正则化右端项: K^T * b
  VectorXd b_regular(N + 1);
  cblas_dgemv(CblasColMajor, CblasTrans, N + 1, N + 1, 1.0, K.data(), N + 1,
              b.data(), 1, 0.0, b_regular.data(), 1);

  // // 若使用 Eigen 库运算
  // // 计算 K^T * K
  // MatrixXd KtK = K.transpose() * K;

  // // 计算 alpha * I + K^T * K
  // MatrixXd alphaI = MatrixXd::Identity(N + 1, N + 1) * alpha;
  // MatrixXd K_regular = alphaI + KtK;

  // // 计算 K^T * y
  // VectorXd b_regular = K.transpose() * b;

  // // 进行求解
  // MatrixXd x = K_regular.llt().solve(b_regular);

  // 定义用于误差计算的精确解
  VectorXd x_e(N + 1);
  for (int i = 0; i < N + 1; i++) {
    x_e(i) = x_exact(i * h);
  }

  // 使用并行 LU 分解求解器求解线性系统
  VectorXd x = parallel_lu_solve(K_regular, b_regular, MPI_COMM_WORLD);

  if (world_rank == 0) {
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
  }

  MPI_Finalize();
  return 0;
}

VectorXd parallel_lu_solve(const MatrixXd &A_global, const VectorXd &b_global,
                           MPI_Comm comm) {

  const double EPSILON = 1e-12;
  int world_size, world_rank;
  MPI_Comm_size(comm, &world_size);
  MPI_Comm_rank(comm, &world_rank);

  const int n = A_global.rows();

  // 各进程确定自己的本地数据布局
  int num_local_cols = n / world_size + (n % world_size > world_rank ? 1 : 0);
  MatrixXd local_A(n, num_local_cols);
  vector<int> global_col_indices(num_local_cols);

  // 各进程自行计算其本地列对应的全局索引
  int current_local_idx = 0;
  for (int j = 0; j < n; ++j) {
    if ((j % world_size) == world_rank) {
      if (current_local_idx < num_local_cols) {
        global_col_indices[current_local_idx++] = j;
      }
    }
  }

  // 使用 MPI_Scatterv 分发矩阵 A。根进程需要重排数据。
  vector<int> sendcounts;
  vector<int> displs;
  MatrixXd temp_A;

  if (world_rank == 0) {
    sendcounts.resize(world_size);
    displs.resize(world_size);
    temp_A.resize(n, n); // 用于存储重排后列的临时矩阵

    int current_col_in_temp = 0;
    // 为 Scatterv 准备 sendcounts, displs 和重排后的数据
    for (int p = 0; p < world_size; ++p) {
      int cols_for_p = n / world_size + (n % world_size > p ? 1 : 0);
      sendcounts[p] = cols_for_p * n; // 发送的元素数量
      displs[p] = (p == 0) ? 0 : displs[p - 1] + sendcounts[p - 1];

      // 将属于进程 p 的列复制到临时矩阵中
      for (int j = 0; j < n; ++j) {
        if ((j % world_size) == p) {
          temp_A.col(current_col_in_temp++) = A_global.col(j);
        }
      }
    }
  }

  MPI_Scatterv(temp_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
               local_A.data(), num_local_cols * n, MPI_DOUBLE, 0, comm);

  // 广播 b 向量
  VectorXd b(n);
  if (world_rank == 0) {
    b = b_global;
  }
  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, comm);

  // 并行 LU 分解 (带部分主元选择)
  vector<double> f_buffer(n);
  for (int j = 0; j < n - 1; ++j) {
    int pivot_owner = j % world_size;
    int l = j;

    // 拥有主元列的进程寻找主元行
    if (world_rank == pivot_owner) {
      int j_local = -1;
      for (int k = 0; k < num_local_cols; ++k) {
        if (global_col_indices[k] == j) {
          j_local = k;
          break;
        }
      }

      double max_val = abs(local_A(j, j_local));
      for (int i = j + 1; i < n; ++i) {
        if (abs(local_A(i, j_local)) > max_val) {
          max_val = abs(local_A(i, j_local));
          l = i;
        }
      }

      if (abs(local_A(l, j_local)) < EPSILON) {
        cerr << "进程 " << world_rank << " 错误: 矩阵在第 " << j
             << " 步奇异或接近奇异。" << endl;
        MPI_Abort(comm, 1);
      }
    }

    // 广播主元行索引
    MPI_Bcast(&l, 1, MPI_INT, pivot_owner, comm);

    // 所有进程执行行交换
    if (l != j) {
      local_A.row(j).swap(local_A.row(l));
      swap(b(j), b(l));
    }
    MPI_Barrier(comm);

    // 主元拥有者计算并广播乘子
    if (world_rank == pivot_owner) {
      int j_local = -1;
      for (int k = 0; k < num_local_cols; ++k) {
        if (global_col_indices[k] == j) {
          j_local = k;
          break;
        }
      }
      for (int i = j + 1; i < n; ++i) {
        local_A(i, j_local) /= local_A(j, j_local);
        f_buffer[i] = local_A(i, j_local);
      }
    }
    MPI_Bcast(f_buffer.data() + j + 1, n - 1 - j, MPI_DOUBLE, pivot_owner,
              comm);

    // 所有进程更新自己的本地数据
    for (int i = j + 1; i < n; ++i) {
      b(i) -= f_buffer[i] * b(j);
    }

    int start_update_col_idx = 0;
    while (start_update_col_idx < num_local_cols &&
           global_col_indices[start_update_col_idx] <= j) {
      start_update_col_idx++;
    }

    for (int k_local = start_update_col_idx; k_local < num_local_cols;
         ++k_local) {
      for (int i = j + 1; i < n; ++i) {
        local_A(i, k_local) -= f_buffer[i] * local_A(j, k_local);
      }
    }
  }

  // 并行回代求解
  VectorXd x = VectorXd::Zero(n);
  for (int i = n - 1; i >= 0; --i) {
    int owner_rank = i % world_size;
    double local_sum = 0.0;

    for (int k_local = 0; k_local < num_local_cols; ++k_local) {
      int j_global = global_col_indices[k_local];
      if (j_global > i) {
        local_sum += local_A(i, k_local) * x(j_global);
      }
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, owner_rank,
               comm);

    double x_i = 0.0;
    if (world_rank == owner_rank) {
      int i_local = -1;
      for (int k = 0; k < num_local_cols; ++k) {
        if (global_col_indices[k] == i) {
          i_local = k;
          break;
        }
      }

      double divisor = local_A(i, i_local);
      if (abs(divisor) < EPSILON) {
        cerr << "进程 " << world_rank << " 错误: 回代时除数为零，i=" << i << "."
             << endl;
        MPI_Abort(comm, 1);
      }
      x_i = (b(i) - global_sum) / divisor;
      x(i) = x_i;
    }

    MPI_Bcast(&x_i, 1, MPI_DOUBLE, owner_rank, comm);
    if (world_rank != owner_rank) {
      x(i) = x_i;
    }
  }

  MPI_Barrier(comm);
  return x;
}