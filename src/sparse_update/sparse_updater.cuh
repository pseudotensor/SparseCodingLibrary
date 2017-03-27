#pragma once
#include <scl.h>
#include "../matrix.cuh"
#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include "../metric.cuh"
#include "cusparse.h"

namespace scl
{
	class SparseUpdater
	{
	public:
		static SparseUpdater* create(const char* dict_updater);

		virtual ~SparseUpdater()
		{
		}

		virtual void init(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) = 0;
		virtual void update(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) = 0;
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

	inline scl_float count_nonzero(Matrix<scl_float>& S)
	{
		//Count non-zeros
		scl_float count = thrust::count_if(S.dptr(), S.dptr() + S.size(), [=]__device__(scl_float val)
		                               {
			                               return val != 0.0f;
		                               });

		return count / S.columns();
	}

	void mp_argmax_columns(const Matrix<scl_float>& M, thrust::device_vector<int>& column_segments, thrust::device_vector<cub::KeyValuePair<int, scl_float>>& argmax_result, DeviceContext& context);

	void mp_update_S(Matrix<scl_float>& S, const thrust::device_vector<cub::KeyValuePair<int, scl_float>>& columns_argmax);


	class MatchingPursuit : public SparseUpdater
	{
		Matrix<scl_float> RT;
		thrust::device_vector<int> idx;
		thrust::device_vector<scl_float> val;
		thrust::device_vector<int> col_ptr;
		Matrix<scl_float> dot_result;
		thrust::device_vector<cub::KeyValuePair<int, scl_float>> columns_argmax;
		thrust::device_vector<int> column_segments;

		void init(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override;

		void sparse_update_residuals(const Matrix<scl_float>& D, thrust::device_vector<cub::KeyValuePair<int, scl_float>>& columns_argmax, Matrix<scl_float>& R, DeviceContext& context, params param);

		void update(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override;
	};


	class OMP : public SparseUpdater
	{
		int orth_iterations;
		Matrix<scl_float> RT;
		thrust::device_vector<int> idx;
		thrust::device_vector<scl_float> val;
		thrust::device_vector<int> col_ptr;
		Matrix<scl_float> dot_result;
		thrust::device_vector<cub::KeyValuePair<int, scl_float>> columns_argmax;
		thrust::device_vector<int> column_segments;

	public:
		OMP(int orth_iterations) : orth_iterations(orth_iterations)
		{
		}

		void init(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override;

		static void orthogonalize(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, Matrix<scl_float>& S_temp, params param, DeviceContext& context, int current_sparsity, int iterations);

		void sparse_residuals(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& R, DeviceContext& context, params param);

		void update(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override;
	};

	inline void zero_values(Matrix<scl_float>& S, params param)
	{
		Matrix<scl_float> S_copy(S);
		Matrix<scl_float> S_sorted(S.rows(), S.columns());

		thrust::transform(S_copy.dptr(), S_copy.dptr() + S_copy.size(), S_copy.dptr(), [=]__device__(scl_float val)
		                  {
			                  return abs(val);
		                  });

		//Sort
		thrust::device_vector<int> indices(S.size());
		thrust::device_vector<int> indices_out(S.size());
		thrust::sequence(indices.begin(), indices.end());
		auto d_indices = thrust::raw_pointer_cast(indices.data());
		auto d_indices_out = thrust::raw_pointer_cast(indices_out.data());
		thrust::device_vector<int> column_segments(S.columns() + 1);
		auto d_segments = thrust::raw_pointer_cast(column_segments.data());
		generate_column_segments(column_segments, S.rows());
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, S_copy.data(), S_sorted.data(), d_indices, d_indices_out,
		                                                   S.size(), S.columns(), d_segments, d_segments + 1);
		// Allocate temporary storage
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run sorting operation
		cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, S_copy.data(), S_sorted.data(), d_indices, d_indices_out,
		                                                   S.size(), S.columns(), d_segments, d_segments + 1);

		cudaFree(d_temp_storage);

		auto counting = thrust::make_counting_iterator(0);
		auto d_S = S.data();
		auto d_S_sorted = S_sorted.data();
		int m = S.rows();
		int target_sparsity = param.target_sparsity;
		thrust::for_each(counting, counting + S.size(), [=]__device__(int idx)
		                 {
			                 int row = idx % m;
			                 if (row >= target_sparsity)
			                 {
				                 int zero_idx = d_indices_out[idx];
				                 d_S[zero_idx] = 0.0f;
			                 }
		                 });
	}

	class IHT : public SparseUpdater
	{
		void init(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
		}

		void update(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			const scl_float eps = 0.99;
			scl_float min_rmse_change = 0.0001;
			S.zero();
			R.copy(X);

			Matrix<scl_float> dot_result(param.dict_size, param.X_n);

			scl_float last_rmse = FLT_MAX;
			const int max_iterations = 10000;
			for (int i = 0; i < max_iterations; i++)
			{
				multiply(D, R, dot_result, context, true);
				multiply(dot_result, eps / param.target_sparsity, context); //TODO: fold multiplication by scalar into the first multiplication

				//Update sparse vector
				add(S, dot_result, S, context);

				//Keep best values
				zero_values(S, param);


				//Update residuals
				multiply(D, S, R, context);
				subtract(X, R, R, context);

				scl_float rmse = rmse_metric(R);
				//printf("%1.8f\n", rmse);

				if (rmse > last_rmse)
				{
					scl_error("Gradient pursuit learning rate too high");
				}

				if (std::abs(rmse - last_rmse) < min_rmse_change)
				{
					printf("Gradient pursuit took %d iterations\n", i);
					break;
				}

				last_rmse = rmse;
			}
		}
	};
}
