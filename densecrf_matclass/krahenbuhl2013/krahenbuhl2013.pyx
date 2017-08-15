# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np

cdef extern from "include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int) except +
        void set_unary_energy(float*)
        void add_pairwise_energy(float*, float*, int)
        void add_potts_pairwise_energy(float, float*, float*, int)
        void set_objective_weighted_log_likelihood(int*, float*, float)
        float gradient_potts(int, float*, float*, int)
        void map(int, int*)
        int npixels()
        int nlabels()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int npixels, int nlabels):
        self.thisptr = new DenseCRFWrapper(npixels, nlabels)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:, ::1] unary_costs):
        if (unary_costs.shape[0] != self.thisptr.npixels() or
                unary_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid unary_costs shape")

        self.thisptr.set_unary_energy(&unary_costs[0, 0])

    def add_pairwise_energy(self, float[:, ::1] pairwise_costs,
                            float[:, ::1] features):
        if (pairwise_costs.shape[0] != self.thisptr.nlabels() or
                pairwise_costs.shape[1] != self.thisptr.nlabels()):
            raise ValueError("Invalid pairwise_costs shape")
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")

        self.thisptr.add_pairwise_energy(
            &pairwise_costs[0, 0],
            &features[0, 0],
            features.shape[1]
        )

    def add_potts_pairwise_energy(
            self, float potts_weight, float[:, ::1] features, float[::1] kernel_params):
        if (features.shape[0] != self.thisptr.npixels()):
            raise ValueError("Invalid features shape")
        if (features.shape[1] != kernel_params.shape[0]):
            raise ValueError("Invalid kernel_params shape")

        self.thisptr.add_potts_pairwise_energy(
            potts_weight,
            &features[0, 0],
            &kernel_params[0],
            features.shape[1]
        )

    def set_objective_weighted_log_likelihood(
        self, int[::1] gt, float[::1] class_weight, float robust):

        if (gt.shape[0] != self.thisptr.npixels() or gt.ndim != 1):
            raise ValueError("Invalid gt shape")
        if (class_weight.shape[0] != self.thisptr.nlabels() or class_weight.ndim != 1):
            raise ValueError("Invalid class_weight shape")

        self.thisptr.set_objective_weighted_log_likelihood(
            &gt[0], &class_weight[0], robust
        )

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        cdef int[::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0])
        return labels

    def gradient_potts(
        self,
        int n_iters,
        float[::1] potts_grad,
        float[::1] kernel_grad,
        int nfeatures):

        if (kernel_grad.shape[0] != nfeatures or kernel_grad.ndim != 1):
            raise ValueError("Invalid kernel_grad shape")
        if (potts_grad.shape[0] != 1 or potts_grad.ndim != 1):
            raise ValueError("Invalid potts_grad shape")

        loss = self.thisptr.gradient_potts(
            n_iters, &potts_grad[0], &kernel_grad[0], nfeatures)

        return loss
