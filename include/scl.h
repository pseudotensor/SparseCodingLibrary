#pragma once

#ifdef WIN32
#define scl_export __declspec(dllexport)
#else
#define scl_export 
#endif

namespace scl
{
	extern "C"
	{

		typedef float  scl_float;

		struct params
		{
			int X_n;
			int X_m;
			int dict_size;
			int target_sparsity;
			int max_iterations;
			const char* sparse_updater;
			const char* dict_updater;
			const char* metric;
			bool print_time;
		};

		/**
		 * \fn	scl_export void sparse_code(const double * _X, double * _D, double * _S, double * _metric, int* _metric_iterations, params _param);
		 *
		 * \brief	Sparse code.
		 *
		 * \author	Rory
		 * \date	3/1/2017
		 *
		 * \param 		  	_X				  	X matrix with column major storage.
		 * \param [in,out]	_D				  	D matrix with column major storage.
		 * \param [in,out]	_S				  	S Matrix with column major storage.
		 * \param [in,out]	_metric			  	Array of size max_iterations. Contains error metric.
		 * \param [in,out]	_metric_iterations	Pointer to a single integer. Number of valid items in
		 * 										_metric.
		 * \param 		  	_param			  	Variable arguments providing additional information.
		 */

		scl_export void sparse_code(const double * _X, double * _D, double * _S, double * _metric, int* _metric_iterations, params _param);
	}
}
