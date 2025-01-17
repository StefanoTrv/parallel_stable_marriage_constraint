libfzn
===
A small C++ library to simplify the interaction with the MiniZinc system.

Installation
--
You can add *libfzn* to your project using CMake:

1. Copy the *libfzn* folder into your project.
2. Add to your CMakeLists.txt the following instructions:

   ```
   ...
   add_subdirectory(libfzn)
   ...
   target_link_libraries(<target> libfzn)
   ```

Quick start
---
The library provides three classes:

- [Parser.h](./Parser.h) It parses a FlatZinc file to create its Model. It uses the grammar defined in [fzn.peg](./fzn.peg).
- [Model.h](./Model.h) It encodes a FlaZinc file using simple data structures.
- [Printer.h](./Printer.h) It prints a solution in the FlatZinc format.

In [Demo.cpp](./Demo.cpp) there is an example of how to use these classes. You can build it using CMake:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make libfzn-demo
```
and test it with a FlatZinc file:

```
./libfzn-demo ../tests/nqueens.fzn
```
