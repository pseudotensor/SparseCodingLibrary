#pragma once
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <sstream>
#include <thrust/system/cuda/error.h>
#include <cusolver_common.h>

namespace scl
{
#define scl_error(x) error(x, __FILE__, __LINE__);

	inline void error(const char* e, const char* file, int line)
	{
		printf("Error: %s, line %d.\n", file, line);
		printf(e);
		printf("\n");
		exit(-1);
	}

#define scl_check(condition, msg) check(condition, msg, __FILE__, __LINE__);

	inline void check(bool val, const char* e, const char* file, int line)
	{
		if (!val)
		{
			error(e, file, line);
		}
	}


#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

	cudaError_t throw_on_cuda_error(cudaError_t code, const char* file, int line)
	{
		if (code != cudaSuccess)
		{
			std::stringstream ss;
			ss << file << "(" << line << ")";
			std::string file_and_line;
			ss >> file_and_line;
			throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
		}

		return code;
	}

	static const char* cublasGetErrorEnum(cublasStatus_t error)
	{
		switch (error)
		{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		}

		return "<unknown>";
	}

#define safe_cublas(ans) throw_on_cublas_error((ans), __FILE__, __LINE__)

	cublasStatus_t throw_on_cublas_error(cublasStatus_t status, const char* file, int line)
	{
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << cublasGetErrorEnum(status) << " - " << file << "(" << line << ")";
			std::string error_text;
			ss >> error_text;
			throw error_text;
		}

		return status;
	}

#define safe_cusolver(ans) throw_on_cusolver_error((ans), __FILE__, __LINE__)
	cusolverStatus_t throw_on_cusolver_error(cusolverStatus_t status, const char* file, int line)
	{
		if (status != CUSOLVER_STATUS_SUCCESS)
		{
			std::stringstream ss;
			ss << "cusolver error: " << file << "(" << line << ")";
			std::string error_text;
			ss >> error_text;
			throw error_text;
		}

		return status;
	}

	template <typename T>
	void print(thrust::device_vector<T>& v)
	{
		thrust::device_vector<T> h_v = v;
		for (int i = 0; i < h_v.size(); i++)
		{
			std::cout << h_v[i] << " ";
		}
		std::cout << "\n";
	}
}
