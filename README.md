# parallel_stable_marriage_constraint
Parallel CP propagator for the stable marriage constraint with CUDA, integrated in the Minicpp solver.

This branch contains the latest version of the propagator. To see the version described in my master's thesis and presented at CILC-2025, check out the [thesis branch](https://github.com/StefanoTrv/parallel_stable_marriage_constraint/tree/thesis).

The code for the Minicpp solver with the serial and parallel SM constraints is available in the folder [fzn-minicpp](/fzn-minicpp/). To compile the project in a Linux environment, run, inside the folder, the following commands:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

A detailed explanation of this project is available in my master's thesis, [downloadable from ResearchGate](https://www.researchgate.net/publication/391483879_A_GPU-based_Parallel_Propagator_fo_the_Stable_Marriage_Constraint).

The serial version of the constraint is based on the article "[An n-ary Constraint for the Stable Marriage Problem](https://arxiv.org/abs/1308.0183)" by Chris Unsworth and Patrick Prosser.
