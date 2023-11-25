#include <functional>
#include <mutex>

#include "worker.h"


namespace manifoldwrapper {
class Controller {
public:
    Controller();

    void add_worker(int tid, const std::function<void()>& f);

    void join_all();

    void barrier();

private:
    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    static inline manifold::Controller* ctrlwrapper_;
    static inline std::once_flag flag_;
};

manifold::Worker* GetCurrentWorker();
manifold::Worker* GetWorker(int tid);

} // namespace manifoldwrapper