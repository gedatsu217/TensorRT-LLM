#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <iostream>

class Worker {
public:
    Worker(int tid, int gpu_id, const std::function<void()>& f);

    int get_tid();

    int get_gpu_id();

    void join();

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

    static Worker* GetWorkerByIdent(size_t ident);

    static Worker* GetCurrentWorker();

    void join_all();

    void add_idmap(int tid, size_t ident);

    void add_worker(int tid, const std::function<void()>& f);

    ~Controller() {
        std::cout << "[Controller] Instance destroyed" << std::endl;
    }

private:
    Controller(int nr_gpus): nr_gpus_(nr_gpus) {
        std::cout << "[Controller] Instance created for " << nr_gpus_ << " gpus" << std::endl;
    }
    
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;
    
    int nr_gpus_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unordered_map<int, size_t> idmap_;
    static inline std::unique_ptr<Controller> instance_;
    static inline std::once_flag flag_;
};