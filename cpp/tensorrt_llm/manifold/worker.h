#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <condition_variable>
#include <queue>

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace manifold{
class Worker {
public:
    Worker(int tid, int gpu_id, const std::function<void()>& f);

    int get_tid();

    int get_gpu_id();

    void join();

    void send_async(int peer, const void* src, size_t src_size, cudaStream_t stream);
    void recv_async(void* recv_buf, size_t recv_size);

    void send_tensor(torch::Tensor tensor, int dst);
    void recv_tensor(torch::Tensor tensor);
    
    void send(int peer, const void* src, size_t src_size, cudaStream_t stream);
    void recv(int peer, void* recv_buf, size_t recv_size);

    std::mutex mtx__;
    std::condition_variable cv__;
    std::queue<std::pair<void*, size_t>> recv_buf_queue_;

    std::mutex mtx_;
    std::condition_variable cv_;
    void* recv_buf_ptr_;
    size_t recv_buf_size_;

    std::mutex mtx___;
    std::condition_variable cv___;

    bool send_end_;

private:
    int tid_;
    size_t ident_;
    int gpu_id_;
    std::unique_ptr<std::thread> thd_;
};

class Controller {
public:
    static Controller* GetInstance();

    static Worker* GetWorker(int tid);

    static Worker* GetCurrentWorker();

    void join_all();

    void add_idmap(size_t ident, int tid);

    void add_worker(int tid, const std::function<void()>& f);

    void barrier();

    ~Controller() {
        std::cout << "[Controller] Instance destroyed" << std::endl;
    }

private:
    Controller(int nr_gpus): nr_gpus_(nr_gpus) {}
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    static Worker* GetWorkerByIdent(size_t ident);
    void barrier_init();
    
    int nr_gpus_;
    pthread_barrier_t barrier_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unordered_map<size_t, int> idmap_;
    static inline std::unique_ptr<Controller> instance_;
    static inline std::once_flag flag_;
    
};
} // namespace manifold