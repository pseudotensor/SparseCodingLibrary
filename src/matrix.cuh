#pragma once
#include "utils.cuh"
#include "device_context.cuh"
#include "cusolverDn.h"
#include <algorithm>

namespace scl
{
	/**
	 * \class	Matrix
	 *
	 * \brief	Matrix type. Stores data internally in column major format.
	 *
	 * \author	Rory
	 * \date	2/20/2017
	 */

	template <typename T>
	class Matrix
	{
		int _n;
		int _m;

		T* _data;

	public:
		Matrix(int n, int m) : _n(n), _m(m)
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
		}

		Matrix(const Matrix<T>& M) : _n(M.columns()), _m(M.rows())
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
			safe_cuda(cudaMemcpy(_data, M.data(),_n*_m* sizeof(T), cudaMemcpyDeviceToDevice));
		}

		~Matrix()
		{
			safe_cuda(cudaFree(_data));
		}

		T* data()
		{
			return _data;
		}

		const T* data() const
		{
			return _data;
		}

		int rows() const
		{
			return _m;
		}

		int columns() const
		{
			return _n;
		}

		int size() const
		{
			return _n * _m;
		}

		void zero()
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, 0);
		}

		void random(int random_seed = 0)
		{
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + _m * _n,
			                  thrust::device_ptr<T>(_data),
			                  [=]__device__(int idx)
			                  {
				                  thrust::default_random_engine randEng(random_seed);
				                  thrust::uniform_real_distribution<float> uniDist;
				                  randEng.discard(idx);
				                  return uniDist(randEng);
			                  }
			);
		}


		void normalize_columns()
		{
			//Create alias so device Lamba does not dereference this pointer
			int m = _m;

			thrust::device_vector<T> temp(_n * _m);
			thrust::device_vector<T> length_squared(_n);

			auto d_data = thrust::device_ptr<T>(_data);
			thrust::transform(d_data, d_data + _n * _m, temp.begin(), [=]__device__(T val)
			                  {
				                  return val * val;
			                  });


			thrust::device_vector<int> column_segments(_n + 1);
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + column_segments.size(), column_segments.begin(), [=]__device__(int idx)
			                  {
				                  return idx * m;
			                  });

			// Determine temporary device storage requirements
			void* d_temp_storage = NULL;
			size_t temp_storage_bytes = 0;
			auto segments = thrust::raw_pointer_cast(column_segments.data());
			auto sum_in = thrust::raw_pointer_cast(temp.data());
			auto sum_out = thrust::raw_pointer_cast(length_squared.data());
			cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, sum_in, sum_out,
			                                _n, segments, segments + 1);
			// Allocate temporary storage
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, sum_in, sum_out,
			                                _n, segments, segments + 1);

			//Scale
			auto d_length_squared = thrust::raw_pointer_cast(length_squared.data());
			thrust::transform(counting, counting + _n * _m, d_data, [=]__device__(int idx)
			                  {
				                  int col = idx / m;

				                  return d_data[idx] / std::sqrt(d_length_squared[col]);
			                  });

			cudaFree(d_temp_storage);
		}

		void print() const
		{
			thrust::host_vector<float> h_mat(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data + _n * _m));
			for (int i = 0; i < _m; i++)
			{
				for (int j = 0; j < _n; j++)
				{
					printf("%1.2f ", h_mat[j * _m + i]);
				}
				printf("\n");
			}
		}
	};

	/**
	 * \fn	void matrix_mul(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false)
	 *
	 * \brief	Matrix multiplication. AB = C. A or B may be transposed.
	 *
	 * \author	Rory
	 * \date	2/21/2017
	 *
	 */

	void matrix_mul(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false)
	{
		cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

		const float alpha = 1;
		const float beta = 0;

		int m = C.rows();
		int n = C.columns();
		int k = transpose_a ? A.rows() : A.columns();
		int lda = transpose_a ? k : m;
		int ldb = transpose_b ? n : k;
		int ldc = m;

		safe_cublas(cublasSgemm(context.cublas_handle, op_a, op_b, m, n, k, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc));
	}

	/**
	 * \fn	void matrix_sub(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context)
	 *
	 * \brief	Matrix subtraction. A - B = C.
	 *
	 * \author	Rory
	 * \date	2/21/2017
	 *
	 */

	void matrix_sub(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context)
	{
		auto counting = thrust::make_counting_iterator(0);
		const float* d_A = A.data();
		const float* d_B = B.data();
		float* d_C = C.data();
		thrust::for_each(counting, counting + A.rows() * A.columns(), [=]__device__(int idx)
		                 {
			                 d_C[idx] = d_A[idx] - d_B[idx];
		                 });
	}

	//https://stackoverflow.com/questions/28794010/solving-linear-systems-ax-b-with-cuda
	//void linear_solve(const Matrix<float>& A, Matrix<float>& X, const Matrix<float>& B, DeviceContext& context)
	//{
	//	Matrix<float> A_copy(A);
	//	scl_check(A.rows()>= A.columns(),"Linear solve requires m >= n");
	//	int work_size = 0;
	//	safe_cusolver(cusolverDnSgeqrf_bufferSize(context.cusolver_handle, A_copy.rows(), A_copy.columns(), A_copy.data(), A_copy.rows(), &work_size));

	//	thrust::device_vector<float> work(work_size);
	//	float *d_work = thrust::raw_pointer_cast(work.data());

	//	thrust::device_vector<float > tau((std::min)(A.rows(), A.columns()));
	//	float *d_tau = thrust::raw_pointer_cast(tau.data());

	//	thrust::device_vector<int> dev_info(1);
	//	int *d_dev_info = thrust::raw_pointer_cast(dev_info.data());

	//	safe_cusolver(cusolverDnSgeqrf(context.cusolver_handle, A_copy.rows(), A_copy.columns(), A_copy.data(), A_copy.rows(), d_tau, d_work, work_size, d_dev_info));

	//	scl_check(dev_info[0] == 0, "geqrf unsuccessful");
	//}

	void pseudoinverse(const Matrix<float>&A, Matrix<float >&pinvA)
	{
		
	}
}
