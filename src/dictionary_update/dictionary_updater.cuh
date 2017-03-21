#pragma once
#include <scl.h>
#include "../matrix.cuh"
#include "../metric.cuh"

namespace scl
{
	class DictionaryUpdater
	{
	public:
		static DictionaryUpdater* create(const char* dict_updater);

		virtual ~DictionaryUpdater()
		{
		}

		virtual void init(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) = 0;
		virtual void update(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) = 0;
	};

	/**
	 * \fn	void linearly_independent_S(Matrix<float >&S)
	 *
	 * \brief	Add tiny amount of noise to S to make it linearly independent.
	 *
	 * \author	Rory
	 * \date	2/28/2017
	 *
	 */

	void linearly_independent_S(Matrix<float>& S)
	{
		auto counting = thrust::make_counting_iterator(0);
		float eps = 1e-3;
		auto d_S = S.data();
		thrust::for_each(counting, counting + S.size(), [=]__device__(int idx)
		                 {
			                 thrust::default_random_engine rng;
			                 thrust::uniform_real_distribution<float> dist(0, eps);
			                 rng.discard(idx);

			                 d_S[idx] = d_S[idx] + dist(rng);
		                 });
	}


	/**
	 * \fn	void normalize_D_S(Matrix<float>& D, Matrix<float>& D_temp, Matrix<float>& S, Matrix<float>& column_length, const Matrix<float>& ones, DeviceContext& context)
	 *
	 * \brief	Normalize D and scale S appropriately. Also check for null atoms and replace with randomness.
	 *
	 * \author	Rory
	 * \date	3/21/2017
	 *
	 * \param [in,out]	D			 	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	D_temp		 	The temporary.
	 * \param [in,out]	S			 	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	column_length	Length of the column.
	 * \param 		  	ones		 	The ones.
	 * \param [in,out]	context		 	The context.
	 */

	void normalize_D_S(Matrix<float>& D, Matrix<float>& D_temp, Matrix<float>& S, Matrix<float>& column_length, const Matrix<float>& ones, DeviceContext& context)
	{
		thrust::transform(D.dptr(), D.dptr() + D.size(), D_temp.dptr(), sqr_op());
		auto d_column_length = column_length.data();
		auto d_ones = ones.data();
		const float alpha = 1.0f;
		const float beta = 0.0f;
		safe_cublas(cublasSgemv(context.cublas_handle, CUBLAS_OP_T, D.rows(), D.columns(), &alpha,D_temp.data(), D.rows(), d_ones, 1, &beta, d_column_length, 1));

		//auto counting = thrust::make_counting_iterator(0);
		//auto d_D = D.data();
		//thrust::for_each(counting, counting + D.size(), [=]__device__(int idx)
		//                 {
		//					 if ()
		//	                 thrust::default_random_engine rng;
		//	                 thrust::uniform_real_distribution<float> dist(0, eps);
		//	                 rng.discard(idx);

		//	                 d_S[idx] = d_S[idx] + dist(rng);
		//                 });

		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(float val)
		                  {
			                  if (val == 0.0f)
			                  {
				                  return 0.0f;
			                  }

			                  return 1.0f / sqrt(val);
		                  });

		safe_cublas(cublasSdgmm(context.cublas_handle, CUBLAS_SIDE_RIGHT, D.rows(), D.columns(), D.data(), D.rows(), d_column_length, 1, D.data(), D.rows()));


		thrust::transform(column_length.dptr(), column_length.dptr() + column_length.size(), column_length.dptr(), [=]__device__(float val)
		                  {
			                  if (val == 0.0f)
			                  {
				                  return 0.0f;
			                  }

			                  return 1.0f / val;
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

		void update(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
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
		Matrix<float> D_column_length;
		Matrix<float> D_ones;
		Matrix<float> XT;
		Matrix<float> DT;
		Matrix<float> ST;
		Matrix<float> RT;

		void init(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
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

		void update(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
		{
			transpose(D, DT, context);
			transpose(S, ST, context);

			gradient_descent_solve(ST, DT, XT, RT, context, 0.8);
			transpose(DT, D, context);

			normalize_D_S(D, DT, S, D_column_length, D_ones, context);
		}
	};

	inline void ksvd_update_S(int atom, Matrix<float>& S, Matrix<float>& g)
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

	inline void restrict_R(int atom, Matrix<float>& R, Matrix<float>& S)
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

	inline void zero_atom(int atom, Matrix<float>& D, params param)
	{
		int begin = atom * param.X_m;
		int end = begin + param.X_m;
		thrust::fill(D.dptr() + begin, D.dptr() + end, 0);
	}

	//Fill null dictionary column with noise
	inline void null_column(Matrix<float>& D, int atom)
	{
		int begin = atom * D.rows();
		int end = begin + D.rows();
		float column_max = thrust::reduce(D.dptr() + begin, D.dptr() + end, 0.0f, [=]__device__(float a, float b)
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
				                 thrust::uniform_real_distribution<float> dist;
				                 rng.discard(idx);

				                 d_D[idx] = dist(rng);
			                 });
		}
	}

	class AKSVD : public DictionaryUpdater
	{
	public:
		Matrix<float> D_column_length;
		Matrix<float> D_ones;
		Matrix<float> D_temp;
		Matrix<float> g;

		void init(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
		{
			D_column_length.resize(D.columns(), 1);
			D_ones.resize(D.columns(), 1);
			D_ones.fill(1.0f);
			D_temp.resize(D.rows(), D.columns());
			g.resize(param.X_n, 1);
		}

		void update(const Matrix<float>& X, Matrix<float>& D, Matrix<float>& S, Matrix<float>& R, params param, DeviceContext& context) override
		{
			int iterations = 2;

			for (int atom = 0; atom < param.dict_size; atom++)
			{
				for (int i = 0; i < iterations; i++)
				{
					zero_atom(atom, D, param);
					residual(X, D, S, R, context);
					restrict_R(atom, R, S);
					//d = Eg
					const float alpha = 1.0f;
					const float beta = 0.0;
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
		else
		{
			std::string error_msg = updater + ": Unknown dictionary updater.";
			scl_error(error_msg.c_str());
			return nullptr;
		}
	}
}
