#pragma once
#include <scl.h>
#include "../matrix.cuh"
#include "../metric.cuh"
#include "../sparse_update/sparse_updater.cuh"
#include <thrust/copy.h>

namespace scl
{
	class DictionaryUpdater
	{
	public:
		static DictionaryUpdater* create(const char* dict_updater);

		virtual ~DictionaryUpdater()
		{
		}

		virtual void init(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) = 0;
		virtual void update(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) = 0;
	};

	/**
	 * \fn	void linearly_independent_S(Matrix<scl_float >&S)
	 *
	 * \brief	Add tiny amount of noise to S to make it linearly independent.
	 *
	 * \author	Rory
	 * \date	2/28/2017
	 *
	 */

	void linearly_independent_S(Matrix<scl_float>& S)
	{
		auto counting = thrust::make_counting_iterator(0);
		scl_float eps = 1e-3;
		auto d_S = S.data();
		thrust::for_each(counting, counting + S.size(), [=]__device__(int idx)
		                 {
			                 thrust::default_random_engine rng;
			                 thrust::uniform_real_distribution<scl_float> dist(0, eps);
			                 rng.discard(idx);

			                 d_S[idx] = d_S[idx] + dist(rng);
		                 });
	}


	/**
	 * \fn	void normalize_D_S(Matrix<scl_float>& D, Matrix<scl_float>& D_temp, Matrix<scl_float>& S, Matrix<scl_float>& column_length, const Matrix<scl_float>& ones, DeviceContext& context)
	 *
	 * \brief	Normalize D and scale S appropriately. 
	 *
	 * \author	Rory
	 * \date	3/21/2017
	 *
	 * \param [in,out]	D			 	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	D_temp		 	The temporary.
	 * \param [in,out]	S			 	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	column_length	Length of the column.
	 * \param 		  	ones		 	The ones.
	 * \param [in,out]	context		 	The context.
	 */

	void normalize_D_S(Matrix<scl_float>& D, Matrix<scl_float>& D_temp, Matrix<scl_float>& S, Matrix<scl_float>& column_length, const Matrix<scl_float>& ones, DeviceContext& context)
	{
		thrust::transform(D.dptr(), D.dptr() + D.size(), D_temp.dptr(), sqr_op());
		auto d_column_length = column_length.data();
		auto d_ones = ones.data();
		const scl_float alpha = 1.0f;
		const scl_float beta = 0.0f;
		safe_cublas(cublasSgemv(context.cublas_handle, CUBLAS_OP_T, D.rows(), D.columns(), &alpha,D_temp.data(), D.rows(), d_ones, 1, &beta, d_column_length, 1));

		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(scl_float val)
		                  {
			                  if (val == 0.0)
			                  {
				                  return 0.0;
			                  }

			                  return 1.0 / sqrt(val);
		                  });

		safe_cublas(cublasSdgmm(context.cublas_handle, CUBLAS_SIDE_RIGHT, D.rows(), D.columns(), D.data(), D.rows(), d_column_length, 1, D.data(), D.rows()));


		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(scl_float val)
		                  {
			                  if (val == 0.0)
			                  {
				                  return 0.0;
			                  }

			                  return 1.0 / val;
		                  });

		safe_cublas(cublasSdgmm(context.cublas_handle, CUBLAS_SIDE_LEFT, S.rows(), S.columns(), S.data(), S.rows(), d_column_length, 1, S.data(), S.rows()));
	}

	/*
	class QRDecomposition : public DictionaryUpdater
	{
	public:
		Matrix<float> D_column_length;
		Matrix<float> D_ones;
		Matrix<float> XT;
		Matrix<float> DT;
		Matrix<float> ST;
		Matrix<float> RT;

		void init(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
		{
			scl_check(param.X_n >= param.dict_size, "QR dictionary update requires n >= dictionary size");
			D_column_length.resize(D.columns(), 1);
			D_ones.resize(D.columns(), 1);
			D_ones.fill(1.0f);
			XT.resize(X.columns(), X.rows());
			transpose(X, XT, context);
			DT.resize(D.columns(), D.rows());
			ST.resize(S.columns(), S.rows());
			RT.resize(X.columns(), X.rows());
		}

		void update(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			linearly_independent_S(S);
			transpose(D, DT, context);
			transpose(S, ST, context);

			linear_solve(ST, DT, XT, context);

			transpose(DT, D, context);

			normalize_D_S(D, DT, S, D_column_length, D_ones, context);
		}
	};
	*/

	class GradientDescent : public DictionaryUpdater
	{
	public:
		Matrix<scl_float> D_column_length;
		Matrix<scl_float> D_ones;
		Matrix<scl_float> XT;
		Matrix<scl_float> DT;
		Matrix<scl_float> ST;
		Matrix<scl_float> RT;

		void init(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			D_column_length.resize(D.columns(), 1);
			D_ones.resize(D.columns(), 1);
			D_ones.fill(1.0f);
			XT.resize(X.columns(), X.rows());
			transpose(X, XT, context);
			DT.resize(D.columns(), D.rows());
			ST.resize(S.columns(), S.rows());
			RT.resize(X.columns(), X.rows());
		}

		void update(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			transpose(D, DT, context);
			transpose(S, ST, context);

			gradient_descent_solve(ST, DT, XT, RT, context, 0.8);
			transpose(DT, D, context);

			normalize_D_S(D, DT, S, D_column_length, D_ones, context);
		}
	};

	class AGD : public DictionaryUpdater
	{
	public:
		Matrix<scl_float> D_column_length;
		Matrix<scl_float> D_ones;
		Matrix<scl_float> DT;
		Matrix<scl_float> S_temp;

		void init(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			D_column_length.resize(D.columns(), 1);
			D_ones.resize(D.columns(), 1);
			D_ones.fill(1.0f);
			DT.resize(D.columns(), D.rows());
			S_temp.resize(S.rows(), S.columns());
		}

		void update(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			residual(X, D, S, R, context);

			const int max_iterations = 1000;
			scl_float best_rmse = FLT_MAX;
			scl_float eps = 0.1;
			scl_float min_rmse_change = 1e-5;

			for (int i = 0; i < max_iterations; i++)
			{
				const scl_float alpha = 1.0;
				const scl_float beta = 0.0f;
				safe_cublas(cublasSgemm(context.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, S.rows(), param.X_m, S.columns(), &alpha, S.data(), S.rows(), R.data(), R.rows(), &beta, DT.data(), DT.rows()));

				scl_float alpha2 = 1.0;
				scl_float beta2 = eps / param.X_m;
				safe_cublas(cublasSgeam(context.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, D.rows(), D.columns(), &alpha2, D.data(), D.rows(), &beta2, DT.data(), DT.rows(), D.data(), D.rows()));

				OMP::orthogonalize(X, D, S, R, S_temp, param, context, param.target_sparsity, 1);

				//Recalculate residual
				residual(X, D, S, R, context);

				scl_float rmse = rmse_metric(R);

				if (std::abs(best_rmse - rmse) < min_rmse_change)
				{
					printf("agd iterations: %d\n",i);
					break;
				}

				if (rmse > best_rmse)
				{
					eps *= 0.5;
				}
				else
				{
					best_rmse = rmse;
					eps *= 1.05;
				}
			}
			normalize_D_S(D, DT, S, D_column_length, D_ones, context);
		}
	};

	inline void ksvd_update_S(int atom, Matrix<scl_float>& S, Matrix<scl_float>& g)
	{
		auto counting = thrust::make_counting_iterator(0);
		auto d_S = S.data();
		auto d_g = g.data();
		int dict_size = S.rows();
		thrust::for_each(counting, counting + g.size(), [=]__device__(int idx)
		                 {
			                 int S_idx = atom + idx * dict_size;
			                 if (d_S[S_idx] != 0)
			                 {
				                 d_S[S_idx] = d_g[idx];
			                 }
		                 });
	}

	inline void restrict_R(int atom, Matrix<scl_float>& R, Matrix<scl_float>& S)
	{
		auto counting = thrust::make_counting_iterator(0);
		auto d_R = R.data();
		auto d_S = S.data();
		int dict_size = S.rows();
		int m = R.rows();
		thrust::for_each(counting, counting + R.size(), [=]__device__(int idx)
		                 {
			                 int column = idx / m;
			                 if (d_S[column * dict_size + atom] == 0)
			                 {
				                 d_R[idx] = 0.0f;
			                 }
		                 });
	}

	inline void zero_atom(int atom, Matrix<scl_float>& D, params param)
	{
		int begin = atom * param.X_m;
		int end = begin + param.X_m;
		thrust::fill(D.dptr() + begin, D.dptr() + end, 0);
	}

	//Fill null dictionary column with noise
	inline void null_column(Matrix<scl_float>& D, int atom)
	{
		int begin = atom * D.rows();
		int end = begin + D.rows();
		scl_float column_max = thrust::reduce(D.dptr() + begin, D.dptr() + end, 0.0f, [=]__device__(scl_float a, scl_float b)
		                                      {
			                                      return max(abs(a), abs(b));
		                                      });

		if (column_max == 0.0f)
		{
			auto counting = thrust::make_counting_iterator(0);
			auto d_D = D.data();
			thrust::for_each(counting + begin, counting + end, [=]__device__(int idx)
			                 {
				                 thrust::default_random_engine rng;
				                 thrust::uniform_real_distribution<scl_float> dist;
				                 rng.discard(idx);

				                 d_D[idx] = dist(rng);
			                 });
		}
	}

	int create_XI_SI(int atom, const Matrix<scl_float>& X, Matrix<scl_float>& XI, const Matrix<scl_float>& S, Matrix<scl_float>& SI, DeviceContext& context)
	{
		auto counting = thrust::make_counting_iterator(0);
		auto d_S = S.data();
		int dict_size = S.rows();
		int m = X.rows();
		auto XI_end = thrust::copy_if(X.dptr(), X.dptr() + X.size(), counting, XI.dptr(), [=]__device__ (int idx)
		                              {
			                              int column = idx / m;
			                              return d_S[column * dict_size + atom] == 0;
		                              });

		thrust::copy_if(S.dptr(), S.dptr() + S.size(), counting, SI.dptr(), [=]__device__ (int idx)
		                {
			                int column = idx / m;
			                return d_S[column * dict_size + atom] == 0;
		                });

		return 0;
	}

	class AKSVD : public DictionaryUpdater
	{
	public:
		Matrix<scl_float> D_column_length;
		Matrix<scl_float> D_ones;
		Matrix<scl_float> D_temp;
		Matrix<scl_float> g;
		Matrix<scl_float> E;
		Matrix<scl_float> XI;
		Matrix<scl_float> SI;

		void init(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			D_column_length.resize(D.columns(), 1);
			D_ones.resize(D.columns(), 1);
			D_ones.fill(1.0f);
			D_temp.resize(D.rows(), D.columns());
			g.resize(param.X_n, 1);
			E.resize(X.rows(), X.columns());
			XI.resize(X.rows(), X.columns());
			SI.resize(S.rows(), S.columns());
		}

		void update(const Matrix<scl_float>& X, Matrix<scl_float>& D, Matrix<scl_float>& S, Matrix<scl_float>& R, params param, DeviceContext& context) override
		{
			int iterations = 2;

			for (int atom = 0; atom < param.dict_size; atom++)
			{
				for (int i = 0; i < iterations; i++)
				{
					zero_atom(atom, D, param);
					residual(X, D, S, R, context);
					//restrict_R(atom, R, S);
					//d = Eg
					const scl_float alpha = 1.0f;
					const scl_float beta = 0.0;
					safe_cublas(cublasSgemv(context.cublas_handle, CUBLAS_OP_N, R.rows(), R.columns(), &alpha, R.data(), R.rows(), S.data() + atom, param.dict_size, &beta, D.data() + atom*D.rows(), 1));


					null_column(D, atom);

					normalize_columns(D, D_temp, D_column_length, D_ones, context);

					//g = Etd
					safe_cublas(cublasSgemv(context.cublas_handle, CUBLAS_OP_T, R.rows(), R.columns(), &alpha, R.data(), R.rows(), D.data() + atom*D.rows(), 1, &beta, g.data(), 1));

					ksvd_update_S(atom, S, g);
				}
			}
		}
	};

	DictionaryUpdater* DictionaryUpdater::create(const char* dict_updater)
	{
		std::string updater(dict_updater);

		if (updater == "gd")
		{
			return new GradientDescent();
		}
		else if (updater == "aksvd")
		{
			return new AKSVD();
		}
		else if (updater == "agd")
		{
			return new AGD();
		}
		else
		{
			std::string error_msg = updater + ": Unknown dictionary updater.";
			scl_error(error_msg.c_str());
			return nullptr;
		}
	}
}
