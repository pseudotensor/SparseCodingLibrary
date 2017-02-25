#pragma once
#include "cublas_v2.h"
#include "utils.cuh"
#include <cusolverDn.h>

namespace scl
{
	class DeviceContext
	{
	public:
		cublasHandle_t cublas_handle;
		cusolverDnHandle_t cusolver_handle;

		DeviceContext()
		{
			safe_cublas(cublasCreate(&cublas_handle));
			safe_cusolver(cusolverDnCreate(&cusolver_handle));
		}

		~DeviceContext()
		{
			safe_cublas(cublasDestroy(cublas_handle));
			safe_cusolver(cusolverDnDestroy(cusolver_handle));
		}
	};
}
