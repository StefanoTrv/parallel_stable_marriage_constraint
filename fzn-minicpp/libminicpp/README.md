# libminicpp
A small C++ library to solve constraint problems.

Installation
--
You can add *libminicpp* to your project using CMake:

1. Copy the *libminicpp* folder into your project.
2. Add to your CMakeLists.txt the following instructions:

   ```
   ...
   add_subdirectory(libminicpp)
   ...
   target_link_libraries(<target> libminicpp)
   ```
   
Quick start
---
The library provides the [MiniCP][minicp] API originally introduced in [HADDOCK][haddock].

In [Demo.cpp](./Demo.cpp) there is an example of how to solve the N-Queens problem. You can build it using CMake:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make minicpp-demo
```
and test it with an arbitrary board size:

```
./minicpp-demo 8
```

[minicp]: https://doi.org/10.1007/s12532-020-00190-7 "MiniCP: A lightweight solver for constraint programming"
[haddock]: https://doi.org/10.1007/978-3-030-58475-7_31 "HADDOCK: A Language and Architecture for Decision Diagram Compilation"