#include "sparse_updater.cuh"

namespace scl
{
	SparseUpdater* SparseUpdater::create(const char* dict_updater)
	{
		std::string updater(dict_updater);

		if (updater == "mp")
		{
			return new MatchingPursuit();
		}
		else if (updater == "iht")
		{
			return new IHT();
		}
		else if (updater == "omp2")
		{
			return new OMP(2);
		}
		else if (updater == "omp5")
		{
			return new OMP(5);
		}
		else if (updater == "omp10")
		{
			return new OMP(10);
		}
		else
		{
			std::string error_msg = updater + ": Unknown sparse updater.";
			scl_error(error_msg.c_str());
			return nullptr;
		}
	}
}
