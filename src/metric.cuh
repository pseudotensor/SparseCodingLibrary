#pragma once
#include "matrix.cuh"


namespace scl
{
	float pyksvd_metric(const Matrix<float>& R);

	float rmse_metric(const Matrix<float>& R);
}
