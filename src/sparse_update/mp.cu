#include "sparse_updater.cuh"
#include "cuda_runtime.h"
#include "../utils.cuh"

namespace scl
{
	void mp_update_S(Matrix<scl_float>& S, const thrust::device_vector<cub::KeyValuePair<int, scl_float>>& columns_argmax)
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
	
	void copy_argmax(const thrust::device_vector<cub::KeyValuePair<int, scl_float>>& columns_argmax, thrust::device_vector<int>& idx, thrust::device_vector<scl_float>& val)
	{
		auto d_idx = thrust::raw_pointer_cast(idx.data());
		auto d_val = thrust::raw_pointer_cast(val.data());
		auto d_columns_argmax = thrust::raw_pointer_cast(columns_argmax.data());

		auto counting = thrust::make_counting_iterator(0);
		thrust::for_each(counting, counting + idx.size(), [=]__device__ (int i)
		                 {
			                 d_val[i] = d_columns_argmax[i].value;
			                 d_idx[i] = d_columns_argmax[i].key;
		                 });
	}

	void mp_argmax_columns(const Matrix<scl_float>& M, thrust::device_vector<int>& column_segments, thrust::device_vector<cub::KeyValuePair<int, scl_float>>& argmax_result, DeviceContext& context)
	{
		typedef cub::ArgIndexInputIterator<const scl_float*, int, scl_float> IterT;
		IterT d_indexed_in(M.data());
		cub::KeyValuePair<int, scl_float> initial_value(1, 0);

		// Determine temporary device storage requirements
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		auto d_segments = thrust::raw_pointer_cast(column_segments.data());
		auto d_argmax_result = thrust::raw_pointer_cast(argmax_result.data());
		cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_indexed_in, d_argmax_result,
		                                   M.columns(), d_segments, d_segments + 1, AbsArgMaxOp(), initial_value);

		context.cub_mem.LazyAllocate(temp_storage_bytes);

		cub::DeviceSegmentedReduce::Reduce(context.cub_mem.d_temp_storage, context.cub_mem.temp_storage_bytes, d_indexed_in, d_argmax_result,
		                                   M.columns(), d_segments, d_segments + 1, AbsArgMaxOp(), initial_value);
	}

	void MatchingPursuit::init(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context)
	{
		idx.resize(param.X_n);
		val.resize(param.X_n);
		col_ptr.resize(param.X_n + 1);
		thrust::sequence(col_ptr.begin(), col_ptr.end(), 0);
		RT.resize(R.columns(), R.rows());
		dot_result.resize(param.dict_size, param.X_n);
		columns_argmax.resize(param.X_n);
		column_segments.resize(param.X_n + 1);
		generate_column_segments(column_segments, dot_result.rows());
	}

	void MatchingPursuit::sparse_update_residuals(const Matrix<scl_float>& D, thrust::device_vector<cub::KeyValuePair<int, scl_float>>& columns_argmax, Matrix<scl_float>& R, DeviceContext& context, params param)
	{
		cusparseMatDescr_t descr = 0;
		const scl_float alpha = 1.0f;
		const scl_float beta = 0.0f;
		auto d_idx = thrust::raw_pointer_cast(idx.data());
		auto d_val = thrust::raw_pointer_cast(val.data());

		copy_argmax(columns_argmax, idx, val);

		auto d_col_ptr = thrust::raw_pointer_cast(col_ptr.data());
		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
		safe_cusparse(cusparseScsrmm2(context.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, param.X_n, param.X_m, param.dict_size, param.X_n, &alpha, descr, d_val, d_col_ptr, d_idx, D.data(), param.X_m, &beta, RT.data(), param.X_n));

		const scl_float alpha2 = -1.0f;
		const scl_float beta2 = 1.0f;
		safe_cublas(cublasSgeam(context.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, R.rows(), R.columns(), &alpha2, RT.data(), RT.rows(), &beta2, R.data(), R.rows(), R.data(), R.rows()));
	}

	void MatchingPursuit::update(const Matrix<scl_float>& X, const Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context)
	{
		S.zero();
		R.copy(X);

		for (int i = 0; i < param.target_sparsity; i++)
		{
			multiply(D, R, dot_result, context, true);

			Timer t;
			mp_argmax_columns(dot_result, column_segments, columns_argmax, context);
			//Update sparse vector
			mp_update_S(S, columns_argmax);

			sparse_update_residuals(D, columns_argmax, R, context, param);
		}
	}
}
