#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <c10/cuda/CUDAStream.h>

#include "workerBinding.h"

namespace py = pybind11;

ControllerWrapper::ControllerWrapper() {
    std::call_once(flag_, []() {
        ctrlwrapper_ = Controller::GetInstance();
    });
}

Worker* ControllerWrapper::GetWorker(int tid) {
    return Controller::GetWorker(tid);
}

Worker* ControllerWrapper::GetCurrentWorker() {
    return Controller::GetCurrentWorker();
}

void ControllerWrapper::add_worker(int tid, const std::function<void()>& f) {
    py::gil_scoped_acquire acquire;
    ctrlwrapper_->add_worker(tid, f);
}

void ControllerWrapper::join_all() {
    py::gil_scoped_release release;
    ctrlwrapper_->join_all();
}

void ControllerWrapper::barrier() {
    //py::gil_scoped_release release;
    ctrlwrapper_->barrier();
    //py::gil_scoped_acquire acquire;
}


void ControllerWrapper::send_tensor(torch::Tensor tensor, int dst) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    //auto ptr = tensor.data_ptr();
    auto tmp = reinterpret_cast<std::int8_t*>(tensor.data_ptr());
    auto ptr = reinterpret_cast<std::uint8_t*>(tmp);
    size_t size = tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
    //printf("[Manifold] Sending tensor of size %lu\n", size);
    auto worker = Controller::GetCurrentWorker();
    //printf("send_tensor: %d to %d\n", worker->get_tid(), dst);
    worker->send_async(dst, ptr, size, stream);
    //worker->send(dst, ptr, worker->get_tid(), size, stream);
}

void ControllerWrapper::recv_tensor(torch::Tensor tensor) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto ptr = tensor.data_ptr();
    size_t size = tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
    //printf("[Manifold] Receiving tensor of size %lu\n", size);
    auto worker = Controller::GetCurrentWorker();
    //printf("recv_tensor: %d\n", worker->get_tid());
    worker->recv_async(ptr, size, stream);
    //worker->recv(ptr, size, stream);
}

int ControllerWrapper::get_current_tid() {
    auto worker = Controller::GetCurrentWorker();
    return worker->get_tid();
}

int ControllerWrapper::get_current_gpu_id() {
    auto worker = Controller::GetCurrentWorker();
    return worker->get_gpu_id();
}

PYBIND11_MODULE(manifoldwrapper, m) {
    py::class_<ControllerWrapper>(m, "ControllerWrapper")
        .def(py::init<>())
        //.def_static("GetWorker", &ControllerWrapper::GetWorker, py::return_value_policy::reference)
        //.def_static("GetCurrentWorker", &ControllerWrapper::GetCurrentWorker, py::return_value_policy::reference)
        .def("join_all", &ControllerWrapper::join_all)  
        .def("add_worker", &ControllerWrapper::add_worker)
        .def("barrier", &ControllerWrapper::barrier, py::call_guard<py::gil_scoped_release>())
        .def("send_tensor", &ControllerWrapper::send_tensor, py::call_guard<py::gil_scoped_release>())
        .def("recv_tensor", &ControllerWrapper::recv_tensor, py::call_guard<py::gil_scoped_release>())
        .def("get_current_tid", &ControllerWrapper::get_current_tid)
        .def("get_current_gpu_id", &ControllerWrapper::get_current_gpu_id);

    /*
    py::class_<Worker>(m, "Worker")
        .def(py::init<int, int, const std::function<void()>&>())
        .def("get_tid", &Worker::get_tid)
        .def("get_gpu_id", &Worker::get_gpu_id)
        .def("join", &Worker::join);
        //.def("send_async", &Worker::send_async)
        //.def("recv_async", &Worker::recv_async)
        //.def("send_tensor", &Worker::send_tensor)
        //.def("recv_tensor", &Worker::recv_tensor);
    */
}
