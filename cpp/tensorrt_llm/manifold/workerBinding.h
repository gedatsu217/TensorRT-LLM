#include <functional>
#include <memory>
#include <mutex>

#include "worker.h"
#include <torch/torch.h>
#include <torch/extension.h>

class ControllerWrapper {
public:
    ControllerWrapper();

    static Worker* GetWorker(int tid);

    static Worker* GetCurrentWorker();

    void add_worker(int tid, const std::function<void()>& f);

    void join_all();

    void barrier();

    void send_tensor(torch::Tensor tensor, int dst);

    void recv_tensor(torch::Tensor tensor);

    int get_current_tid();

    int get_current_gpu_id();

private:
    ControllerWrapper(const ControllerWrapper&) = delete;
    ControllerWrapper& operator=(const ControllerWrapper&) = delete;

    static inline Controller* ctrlwrapper_;
    static inline std::once_flag flag_;
};