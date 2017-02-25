#include <cstdio>
#include "cuda_runtime.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <cub/cub.cuh>
#include "utils.cuh"
#include "matrix.cuh"
#include "device_context.cuh"

namespace std
{
	class slice;
}

using namespace scl;

struct params
{
	params(int X_n, int X_m, int dict_size, int target_sparsity)
		: X_n(X_n),
		  X_m(X_m),
		  dict_size(dict_size),
		  target_sparsity(target_sparsity)
	{
	}

	int X_n;
	int X_m;
	int dict_size;
	int target_sparsity;
};

struct AbsArgMaxOp
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T& a, const T& b) const
	{
		return (std::abs(b.value) < std::abs(a.value)) ? a : b;
	}
};

void argmax_columns(const Matrix<float>& M, thrust::device_vector<int>& column_segments, thrust::device_vector<cub::KeyValuePair<int, float>>& argmax_result, DeviceContext& context)
{
	typedef cub::ArgIndexInputIterator<const float*, int, float> IterT;
	IterT d_indexed_in(M.data());
	cub::KeyValuePair<int, float> initial_value(1, 0);

	// Determine temporary device storage requirements
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	auto d_segments = thrust::raw_pointer_cast(column_segments.data());
	auto d_argmax_result = thrust::raw_pointer_cast(argmax_result.data());
	cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_indexed_in, d_argmax_result,
	                                   M.columns(), d_segments, d_segments + 1, AbsArgMaxOp(), initial_value);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_indexed_in, d_argmax_result,
	                                   M.columns(), d_segments, d_segments + 1, AbsArgMaxOp(), initial_value);

	cudaFree(d_temp_storage);
}

void update_S(Matrix<float>& S, const thrust::device_vector<cub::KeyValuePair<int, float>>& columns_argmax)
{
	auto counting = thrust::make_counting_iterator(0);
	auto d_S = thrust::raw_pointer_cast(S.data());
	auto d_argmax = thrust::raw_pointer_cast(columns_argmax.data());
	int m = S.rows();
	thrust::for_each(counting, counting + columns_argmax.size(), [=]__device__(int idx)
	                 {
		                 auto kv = d_argmax[idx];
		                 d_S[idx * m + kv.key] += kv.value;
	                 });
}

template <typename T>
struct absolute_value : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T& x) const
	{
		return x < T(0) ? -x : x;
	}
};


float calculate_RMSE(const Matrix<float>& R)
{
	auto dptr = thrust::device_pointer_cast(R.data());
	auto sqr = [=]__device__(float val)
		{
			return val * val;
		};

	float MSE = thrust::transform_reduce(dptr, dptr + R.size(), sqr,
	                                     0.0f, thrust::plus<float>()) / R.size();
	return std::sqrt(MSE);
}

void matching_pursuit(const Matrix<float>& X, const Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context)
{
	S.zero();

	Matrix<float> dot_result(param.X_n, param.dict_size);

	thrust::device_vector<cub::KeyValuePair<int, float>> columns_argmax(dot_result.columns());

	thrust::device_vector<int> column_segments(dot_result.columns() + 1);

	auto counting = thrust::make_counting_iterator(0);
	int dict_size = param.dict_size;
	thrust::transform(counting, counting + column_segments.size(), column_segments.begin(), [=]__device__(int idx)
	                  {
		                  return idx * dict_size;
	                  });


	for (int i = 0; i < param.target_sparsity; i++)
	{
		matrix_mul(D, R, dot_result, context, true);

		printf("Dot result\n");
		dot_result.print();

		argmax_columns(dot_result, column_segments, columns_argmax, context);

		//Update sparse vector
		update_S(S, columns_argmax);

		printf("S\n");
		S.print();

		//Update residuals
		matrix_mul(D, S, R, context);
		matrix_sub(X, R, R, context);

		printf("Residuals:\n");
		R.print();
	}
	printf("RMSE: %1.4f\n", calculate_RMSE(R));
}

void sparse_code(const Matrix<float>& X, params param, DeviceContext& context)
{
	Matrix<float> D(param.dict_size, param.X_m); //Dictionary
	D.random(9);
	D.normalize_columns();

	Matrix<float> S(param.X_n, param.dict_size); //Sparse coding
	Matrix<float> R(X); //Residuals


	printf("D\n");
	D.print();
	printf("X\n");
	X.print();

	matching_pursuit(X, D, S, R, param, context);
}

int main()
{
	DeviceContext context;

	int dict_size = 4;
	int m = 3;
	int n = 10;
	int target_sparsity = 3;

	params param(n, m, dict_size, target_sparsity);

	Matrix<float> X(param.X_n, param.X_m);
	X.random(0);

	sparse_code(X, param, context);

	return 0;
}
