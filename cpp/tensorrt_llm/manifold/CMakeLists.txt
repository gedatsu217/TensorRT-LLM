set(pybind11_DIR /usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

add_library(manifold SHARED worker.cpp)
set_property(TARGET manifold PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(manifold PRIVATE ${CUDA_RUNTIME_LIBRARY} ${TORCH_LIBRARIES})


pybind11_add_module(manifoldwrapper SHARED workerBinding.cpp)
set_property(TARGET manifoldwrapper PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(manifoldwrapper PRIVATE manifold ${TORCH_PYTHON_LIBRARY} ${TORCH_LIBRARIES} ${CUDA_RUNTIME_LIBRARY})
