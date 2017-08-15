#include "densecrf.h"

class DenseCRFWrapper {
	public:
		DenseCRFWrapper(int npixels, int nlabels);
		virtual ~DenseCRFWrapper();

		void set_unary_energy(float* unary_costs_ptr);

		void add_pairwise_energy(float* pairwise_costs_ptr,
				float* features_ptr, int nfeatures);

		void add_potts_pairwise_energy(float potts_weight,
				float* features_ptr, float* kernel_params_ptr,
				int nfeatures);

    void set_objective_weighted_log_likelihood(
        int* gt_ptr, float* weight_ptr, float robust);

		void map(int n_iters, int* result);

    float gradient_potts(int n_iters,
        float* potts_grad_ptr, float* kernel_grad_ptr, int nfeatures);

		int npixels();
		int nlabels();

	private:
		DenseCRF* m_crf;
		ObjectiveFunction *m_objective;
		int m_npixels;
		int m_nlabels;
		int m_num_pairwise;
};
