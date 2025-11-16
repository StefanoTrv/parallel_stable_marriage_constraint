# parallel_stable_marriage_constraint
Parallel CP propagator for the stable marriage constraint with CUDA, integrated in the Minicpp solver.

This branch contains a version of the propagator in which the threads synchronize to launch the propagator's phases from inside the device, instead of letting the host launch new phases.