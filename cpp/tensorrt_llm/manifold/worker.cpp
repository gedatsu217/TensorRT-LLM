#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "worker.h"

namespace manifold{
// Class Worker
Worker::Worker(int tid, int gpu_id, const std::function<void()>& f): tid_(tid), gpu_id_(gpu_id), recv_buf_ptr_(nullptr), recv_buf_size_(0), send_end_(false) {  
    thd_ = std::make_unique<std::thread>([this, f]() {
        auto ident_ = std::hash<std::thread::id>{}(std::this_thread::get_id());
        Controller::GetInstance()->add_idmap(ident_, tid_);
        f();
    });
}

int Worker::get_tid() {
    return tid_;
}

int Worker::get_gpu_id() {
    return gpu_id_;
}

void Worker::recv_async(void* recv_buf, size_t recv_size)
{
    {
        std::lock_guard<std::mutex> lg(mtx__);
        recv_buf_queue_.push({recv_buf, recv_size});  
    }
    cv__.notify_all();
}

void Worker::send_async(int peer, const void* src, size_t src_size, cudaStream_t stream) {
    auto peer_worker = Controller::GetWorker(peer);
    std::pair<void*, size_t> recv_buf_pair;

    {
        std::unique_lock<std::mutex> ul(peer_worker->mtx__);
        peer_worker->cv__.wait(ul, [peer_worker] { return !peer_worker->recv_buf_queue_.empty();});
        recv_buf_pair = peer_worker->recv_buf_queue_.front();
        peer_worker->recv_buf_queue_.pop();
    }
    
    void* recv_buf_ = recv_buf_pair.first;
    size_t recv_size_ = recv_buf_pair.second;
    
    if (src_size > recv_size_) {
        std::cout << "Dst buffer is too small!" << std::endl;
        exit(-1);
    }

    cudaMemcpyPeerAsync(recv_buf_, peer, src, tid_, src_size, stream);
}

void Worker::recv(int peer, void* recv_buf, size_t recv_size) {
    auto peer_worker = Controller::GetWorker(peer);

    {
        std::lock_guard<std::mutex> lg(mtx_);
        recv_buf_ptr_ = recv_buf;
        recv_buf_size_ = recv_size;
    }

    cv_.notify_all();

    {
        std::unique_lock<std::mutex> ul(peer_worker->mtx___);
        peer_worker->cv___.wait(ul, [peer_worker] { return peer_worker->send_end_;});    
        peer_worker->send_end_ = false;
    }
}

void Worker::send(int peer, const void* src, size_t src_size, cudaStream_t stream) {
    auto peer_worker = Controller::GetWorker(peer);
    void* recv_buf_ptr;
    size_t recv_buf_size;

    {
        std::unique_lock<std::mutex> ul(peer_worker->mtx_);
        peer_worker->cv_.wait(ul, [peer_worker] { return peer_worker->recv_buf_ptr_ != nullptr;});
        recv_buf_ptr = peer_worker->recv_buf_ptr_;
        recv_buf_size = peer_worker->recv_buf_size_;
        
        peer_worker->recv_buf_ptr_ = nullptr;
        peer_worker->recv_buf_size_ = 0;
    }
    
    if (src_size > recv_buf_size) {
        std::cout << "Dst buffer is too small!" << std::endl;
        exit(-1);
    }        

    cudaStreamSynchronize(stream); //ToDo: is this necessary? It seems not to be necessary because cudaMemcpyPeer is a sync-function, but it fails without syncronization.
    cudaMemcpyPeerAsync(recv_buf_ptr, peer, src, tid_, src_size, stream);
    cudaStreamSynchronize(stream);

    {
        std::lock_guard<std::mutex> lg(mtx___);
        send_end_ = true;
    }
    cv___.notify_all();
}

void Worker::send_tensor(torch::Tensor tensor, int dst) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto ptr = tensor.data_ptr();
    size_t size = tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
    send_async(dst, ptr, size, stream);
}

void Worker::recv_tensor(torch::Tensor tensor) {
    auto ptr = tensor.data_ptr();
    size_t size = tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
    recv_async(ptr, size);
}

void Worker::join() {
    thd_->join();
}

// Class Controller
//public
Controller* Controller::GetInstance() {
    std::call_once(flag_, []() {
        int nr_gpus;
        cudaGetDeviceCount(&nr_gpus);
        instance_ = std::unique_ptr<Controller>(new Controller(nr_gpus));   
        instance_->barrier_init();
        printf("[Manifold] Controller instance created for %d gpus\n", nr_gpus);
});
    
    return instance_.get();
}

Worker* Controller::GetWorker(int tid) {
    return GetInstance()->workers_[tid].get();
}

Worker* Controller::GetCurrentWorker() {
    auto current_ident = std::this_thread::get_id();
    auto ident = std::hash<std::thread::id>{}(current_ident);
    return GetWorkerByIdent(ident);
}

void Controller::join_all() {
    for (auto& worker : workers_) {
        worker->join();
    }
    printf("[Manifold] All workers joined!\n");
}

void Controller::add_idmap(size_t ident, int tid) { // Public for the Worker constructor, but not exported to Python
    idmap_[ident] = tid;
}

void Controller::add_worker(int tid, const std::function<void()>& f) {
    int gpu_id = (nr_gpus_ == 0) ? 0 : tid % nr_gpus_;
    std::cout << "[Manifold] Adding a worker with tid: " << tid << std::endl;
    workers_.emplace_back(std::make_unique<Worker>(tid, gpu_id, f));
}

void Controller::barrier() {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(stream);
    pthread_barrier_wait(&barrier_);
}

//private: 
Worker* Controller::GetWorkerByIdent(size_t ident) {
    int tid = GetInstance()->idmap_[ident];
    return GetWorker(tid);
}

void Controller::barrier_init() {
    pthread_barrier_init(&barrier_, NULL, nr_gpus_);
}

} // namespace manifold