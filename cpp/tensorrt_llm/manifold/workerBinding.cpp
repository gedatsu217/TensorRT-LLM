#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "workerBinding.h"


namespace py = pybind11;

namespace manifoldwrapper{

Controller::Controller() {
    std::call_once(flag_, []() {
        ctrlwrapper_ = manifold::Controller::GetInstance();
    });
}

void Controller::add_worker(int tid, const std::function<void()>& f) {
    py::gil_scoped_acquire acquire;
    ctrlwrapper_->add_worker(tid, f);
}

void Controller::join_all() {
    py::gil_scoped_release release;
    ctrlwrapper_->join_all();
}

void Controller::barrier() {
    ctrlwrapper_->barrier();
}

manifold::Worker* GetCurrentWorker() {
    return manifold::Controller::GetCurrentWorker();
}

manifold::Worker* GetWorker(int tid) {
    return manifold::Controller::GetWorker(tid);
}

} // namespace manifoldwrapper

PYBIND11_MODULE(manifoldwrapper, m) {
    py::class_<manifoldwrapper::Controller>(m, "Controller")
        .def(py::init<>())
        .def("join_all", &manifoldwrapper::Controller::join_all)
        .def("add_worker", &manifoldwrapper::Controller::add_worker)
        .def("barrier", &manifoldwrapper::Controller::barrier, py::call_guard<py::gil_scoped_release>());
    
    py::class_<manifold::Worker>(m, "Worker")
        .def("get_tid", &manifold::Worker::get_tid)
        .def("get_gpu_id", &manifold::Worker::get_gpu_id)
        .def("send_tensor", &manifold::Worker::send_tensor, py::call_guard<py::gil_scoped_release>())
        .def("recv_tensor", &manifold::Worker::recv_tensor, py::call_guard<py::gil_scoped_release>());
    
    m.def("GetCurrentWorker", &manifoldwrapper::GetCurrentWorker, py::return_value_policy::reference);
    m.def("GetWorker", &manifoldwrapper::GetWorker, py::return_value_policy::reference);
}
