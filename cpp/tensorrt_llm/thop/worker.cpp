#include <thread>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "tensorrt_llm/thop/worker.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace py = pybind11;
namespace th = torch;


Worker::Worker(int tid, int gpu_id, const std::function<void()>& f): tid_(tid), gpu_id_(gpu_id) {  
    thd_ = std::make_unique<std::thread>([this, f]() {
        auto ident_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
        Controller::GetInstance()->add_idmap(tid_, ident_);
        py::gil_scoped_acquire acquire;
        f();
    });
}

int Worker::get_tid() {
    return tid_;
}

int Worker::get_gpu_id() {
    return gpu_id_;
}

void Worker::join() {
    py::gil_scoped_release release;    
    thd_->join();
}


Controller* Controller::GetInstance() {
    std::call_once(flag_, []() {
        int nr_gpus = th::cuda::device_count();
        instance_ = std::unique_ptr<Controller>(new Controller(nr_gpus));   
});
    return instance_.get();
}

Worker* Controller::GetWorker(int tid) {
    return GetInstance()->workers_[tid].get();
}

Worker* Controller::GetWorkerByIdent(size_t ident) {
    int tid = GetInstance()->idmap_[ident];
    return GetWorker(tid);
}

Worker* Controller::GetCurrentWorker() {
    auto ident = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return GetWorkerByIdent(ident);
}

void Controller::join_all() {
    for (auto& worker : workers_) {
        worker->join();
    }
    printf("[Controller] All workers joined!\n");
}

void Controller::add_idmap(int tid, size_t ident) {
    idmap_[tid] = ident;
}

void Controller::add_worker(int tid, const std::function<void()>& f) {
    int gpu_id = (nr_gpus_ == 0) ? 0 : tid % nr_gpus_;
    std::cout << "[Controller] Adding a worker with tid: " << tid << std::endl;
    workers_.emplace_back(std::make_unique<Worker>(tid, gpu_id, f));
}


PYBIND11_MODULE(manifold, m) {
    py::class_<Worker>(m, "Worker")
        .def(py::init<int, int, std::function<void()>&>())
        .def("get_tid", &Worker::get_tid)
        .def("get_gpu_id", &Worker::get_gpu_id)
        .def("join", &Worker::join);
    
    py::class_<Controller>(m, "Controller")
        //.def(py::init(&Controller::GetInstance))  //ToDo?: py::return_value_policy::reference doesn't seem to work properly. For some reason, destructor is called twice???
        .def_static("GetInstance", &Controller::GetInstance, py::return_value_policy::reference)
        .def_static("GetWorker", &Controller::GetWorker)
        .def_static("GetWorkerByIdent", &Controller::GetWorkerByIdent)
        .def_static("GetCurrentWorker", &Controller::GetCurrentWorker)
        .def("join_all", &Controller::join_all)
        .def("add_idmap", &Controller::add_idmap)
        .def("add_worker", &Controller::add_worker);
}



