#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "matrix.cuh"
#include "device_context.cuh"
#include <scl.h>
#include "dictionary_update/dictionary_updater.cuh"
#include "sparse_update/sparse_updater.cuh"
#include <ctime>
#include "metric.cuh"


namespace scl
{
	float calculate_metric(const Matrix<float>& X, const Matrix<float>& D, const Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context)
	{
		//Update residuals
		multiply(D, S, R, context);
		subtract(X, R, R, context);
		std::string metric(param.metric);
		if (metric == "rmse")
		{
			return rmse_metric(R);
		}
		if (metric == "pyksvd")
		{
			return pyksvd_metric(R);
		}
		else
		{
			scl_error(std::string("Unknown metric: " + metric).c_str());
			return 0;
		}
	}

	void random_D(Matrix<float>& D, Matrix<float>& D_temp, DeviceContext& context)
	{
		D.random(7);
		Matrix<float> D_column_length(D.columns(), 1);
		Matrix<float> D_ones(D.columns(), 1);
		D_ones.fill(1.0);
		normalize_columns(D, D_temp, D_column_length, D_ones, context);
	}

	void sparse_code(const float* _X, float* _D, float* _S, float* _metric, int* _metric_iterations, params _param)
	{
		try
		{
			DeviceContext context;

			Matrix<float> X(_param.X_m, _param.X_n);
			X.copy(_X);

			Matrix<float> R(X.rows(), X.columns()); //Residuals

			Matrix<float> D(_param.X_m, _param.dict_size); //Dictionary
			random_D(D, R, context);

			Matrix<float> S(_param.dict_size, _param.X_n); //Sparse coding
			S.zero();

			DictionaryUpdater* dict_updater = DictionaryUpdater::create(_param.dict_updater);
			dict_updater->init(X, D, S, R, _param, context);
			SparseUpdater* sparse_updater = SparseUpdater::create(_param.sparse_updater);
			sparse_updater->init(X, D, S, R, _param, context);

			*_metric_iterations = 0;
			clock_t sparse_time = 0;
			clock_t dict_time = 0;
			for (int i = 0; i < _param.max_iterations; i++)
			{
				clock_t tic = clock();
				sparse_updater->update(X, D, S, R, _param, context);
				safe_cuda(cudaDeviceSynchronize());
				sparse_time += clock() - tic;

				tic = clock();
				dict_updater->update(X, D, S, R, _param, context);
				safe_cuda(cudaDeviceSynchronize());
				dict_time += clock() - tic;

				_metric[i] = calculate_metric(X, D, S, R, _param, context);
				(*_metric_iterations)++;
			}

			if (_param.print_time)
			{
				printf("sparse update: %1.4fs\n", clocks_to_s(sparse_time));
				printf("dictionary update: %1.4fs\n", clocks_to_s(dict_time));
			}

			//Copy D and S back
			thrust::copy(D.dptr(), D.dptr() + D.size(), _D);
			thrust::copy(S.dptr(), S.dptr() + S.size(), _S);
		}
		catch (std::exception e)
		{
			std::cerr << "scl error: " << e.what() << "\n";
		}
		catch (std::string e)
		{
			std::cerr << "scl error: " << e << "\n";
		}
		catch (...)
		{
			std::cerr << "scl error\n";
		}
	}
}
