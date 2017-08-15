#include <Eigen/Core>
#include "densecrf.h"
#include "densecrf_wrapper.h"

DenseCRFWrapper::DenseCRFWrapper(int npixels, int nlabels)
: m_objective(NULL), m_npixels(npixels), m_nlabels(nlabels), m_num_pairwise(0) {
	m_crf = new DenseCRF(npixels, nlabels);
}

DenseCRFWrapper::~DenseCRFWrapper() {
	delete m_crf;
	if (m_objective) delete m_objective;
}

int DenseCRFWrapper::npixels() { return m_npixels; }
int DenseCRFWrapper::nlabels() { return m_nlabels; }

void DenseCRFWrapper::add_pairwise_energy(float* pairwise_costs_ptr, float* features_ptr, int nfeatures) {
	m_crf->addPairwiseEnergy(
		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
		new MatrixCompatibility(
			Eigen::Map<const Eigen::MatrixXf>(pairwise_costs_ptr, m_nlabels, m_nlabels)
		),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC
	);
	m_num_pairwise ++;
}

void DenseCRFWrapper::add_potts_pairwise_energy(float potts_weight, float* features_ptr, float* kernel_params_ptr, int nfeatures) {
	m_crf->addPairwiseEnergy(
		Eigen::Map<const Eigen::MatrixXf>(features_ptr, nfeatures, m_npixels),
		new PottsCompatibility(potts_weight),
		DIAG_KERNEL,
		NORMALIZE_SYMMETRIC
	);
	m_crf->setKernelParameters(Eigen::Map<const Eigen::VectorXf>(kernel_params_ptr, nfeatures));
	m_num_pairwise ++;
}

void DenseCRFWrapper::set_unary_energy(float* unary_costs_ptr) {
	m_crf->setUnaryEnergy(
		Eigen::Map<const Eigen::MatrixXf>(
			unary_costs_ptr, m_nlabels, m_npixels)
	);
}

void DenseCRFWrapper::map(int n_iters, int* labels) {
	VectorXs labels_vec = m_crf->map(n_iters);
	for (int i = 0; i < m_npixels; i ++)
		labels[i] = labels_vec(i);
}

void DenseCRFWrapper::set_objective_weighted_log_likelihood(
    int* gt_ptr, float* class_weight_ptr, float robust) {

  m_objective = new WeightedLogLikelihood(
      Eigen::Map<const Eigen::VectorXi>(gt_ptr, m_npixels).cast<short>(),
      Eigen::Map<const Eigen::VectorXf>(class_weight_ptr, m_nlabels),
      robust
  );
}

float DenseCRFWrapper::gradient_potts(
    int n_iters,
    float* potts_grad_ptr,
    float* kernel_grad_ptr,
    int nfeatures) {

  assert(m_num_pairwise == 1);
  assert(m_objective);

  VectorXf lbl_cmp_grad;
  VectorXf kernel_grad;

  float r = m_crf->gradient(
      n_iters, *m_objective, NULL, &lbl_cmp_grad, &kernel_grad);

  assert(lbl_cmp_grad.rows() == 1);
  assert(kernel_grad.rows() == nfeatures);

  // negate since krahenbuhl2013 computes the negative gradient
  potts_grad_ptr[0] = -lbl_cmp_grad(0);
  for (int i = 0; i < nfeatures; i ++) {
    kernel_grad_ptr[i] = -kernel_grad(i);
  }
  return -r;
}
