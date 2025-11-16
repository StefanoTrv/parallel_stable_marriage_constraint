# parallel_stable_marriage_constraint
Parallel CP propagator for the stable marriage constraint with CUDA, integrated in the Minicpp solver.

This branch contains the an experimental version of the propagator where the phases are launched from inside the device. This is done by having some small kernels that are queued in the stream and executed when the previous main phase is completed. These small kernels prepare the data for the next step and decide which step of the propagator should be lauched next.
