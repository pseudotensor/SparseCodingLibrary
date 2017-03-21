#include "sparse_updater.cuh"
#include "cuda_runtime.h"
#include "../utils.cuh"

namespace scl
{
	void orth_update_S(Matrix<float>& S, Matrix<float>& dot_result)
	{
		auto counting = thrust::make_counting_iterator(0);
		auto d_S = S.data();
		auto d_dot_result = dot_result.data();
		thrust::for_each(counting, counting + S.size(), [=]__device__ (int idx)
		                 {
			                 if (d_S[idx] != 0.0f)
			                 {
				                 d_S[idx] += d_dot_result[idx];
			                 }
		                 });
	}

	void OMP::orthogonalize(const Matrix<float>& X, const Matrix<float>& D, Matrix<float >&S, Matrix<float>& R, Matrix<float>& dot_result, params param, DeviceContext& context, int current_sparsity, int iterations)
	{
		if (current_sparsity == 1)
		{
			return;
		}

		const float eps = 0.5;
		for (int i = 0; i < iterations; i++)
		{
			residual(X, D, S, R, context);

			multiply(D, R, dot_result, context, true, false, eps / current_sparsity);

			orth_update_S(S,dot_result);
		}
	}

	void OMP::init(const Matrix<float>& X, const Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context)
	{
		dot_result.resize(param.dict_size, param.X_n);
		columns_argmax.resize(param.X_n);
		column_segments.resize(param.X_n + 1);
		generate_column_segments(column_segments, dot_result.rows());
	}

	void OMP::update(const Matrix<float>& X, const Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context)
	{
		S.zero();
		R.copy(X);

		for (int i = 0; i < param.target_sparsity; i++)
		{
			multiply(D, R, dot_result, context, true);

			mp_argmax_columns(dot_result, column_segments, columns_argmax, context);

			mp_update_S(S, columns_argmax);

			this->orthogonalize(X, D, S, R, dot_result, param, context, i + 1, orth_iterations);

			residual(X, D, S, R, context);
		}
	}
}
