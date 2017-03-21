#include "metric.cuh"
#include "utils.cuh"
#include "matrix.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

namespace scl
{

	float pyksvd_metric(const Matrix<float>& R)
	{
		auto dptr = thrust::device_pointer_cast(R.data());

		float sum_square = thrust::transform_reduce(dptr, dptr + R.size(), sqr_op(),
		                                            0.0f, thrust::plus<float>());
		float f_norm = std::sqrt(sum_square);
		return f_norm / R.columns();
	}

	float rmse_metric(const Matrix<float>& R)
	{
		auto dptr = thrust::device_pointer_cast(R.data());

		float MSE = thrust::transform_reduce(dptr, dptr + R.size(), sqr_op(),
		                                     0.0f, thrust::plus<float>()) / R.size();
		return std::sqrt(MSE);
	}
}
