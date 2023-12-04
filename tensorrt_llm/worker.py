import torch
import threading
from collections import OrderedDict
from cuda import cudart
import queue

def CUDACHECK(cmd):
    err = cmd
    if err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA command returned error: {}".format(err))
    

class Worker:
    def __init__(self, tid, gpu_id, f):
        self.__tid = tid
        self.__gpu_id = gpu_id
        self.recv_queue = queue.Queue()
        self.cv = threading.Condition()
        def thf():
            ident = threading.get_native_id()
            Controller().add_idmap(ident, tid)
            f()
        self.__thd = threading.Thread(target=thf)
        self.__thd.start()
    
    def get_tid(self):
        return self.__tid
    
    def get_gpu_id(self):
        return self.__gpu_id
    
    def join(self):
        self.__thd.join()

    def send_async(self, peer, src, src_size, stream):
        peer_worker = GetWorker(peer)
        recv_buf = 0
        recv_size = 0
        with peer_worker.cv:
            while peer_worker.recv_queue.empty():
                peer_worker.cv.wait()
            recv_buf, recv_size = peer_worker.recv_queue.get()
        
        if src_size > recv_size:
            raise RuntimeError("send_async: src_size > recv_size")
        
        CUDACHECK(cudart.cudaMemcpyPeerAsync(recv_buf, peer, src, self.__gpu_id, src_size, stream.cuda_stream))
        #CUDACHECK(cudart.cudaMemcpyPeer(recv_buf, peer, src, self.__gpu_id, src_size))
    
    def recv_async(self, recv_buf, recv_size):
        with self.cv:
            self.recv_queue.put((recv_buf, recv_size))
            self.cv.notify_all()

    def send_tensor(self, tensor, dst):
        stream = torch.cuda.current_stream()
        tensor_ptr = tensor.data_ptr()
        tensor_size = tensor.numel() * tensor.element_size()
        self.send_async(dst, tensor_ptr, tensor_size, stream)

    def recv_tensor(self, tensor):
        tensor_ptr = tensor.data_ptr()
        tensor_size = tensor.numel() * tensor.element_size()
        self.recv_async(tensor_ptr, tensor_size)
    

class Controller:
    __instance = None
    __instance_lock = threading.Lock()
    __workers = []
    __worker_lock = threading.Lock()
    __idmap = OrderedDict()
    __idmap_lock = threading.Lock()

    def __new__(cls):
        if cls.__instance is None:
            with cls.__instance_lock:
                if cls.__instance is None:
                    cls.__instance = super(Controller, cls).__new__(cls)
                    nr_gpus = torch.cuda.device_count()
                    cls.__instance.__nr_gpus = nr_gpus
                    cls.__instance.__barrier = threading.Barrier(nr_gpus)
                    print("[Manifold] Controller instance created for {} GPUs".format(cls.__instance.__nr_gpus))
        return cls.__instance
    
    def add_worker(self, tid, f):
        self.__worker_lock.acquire()
        gpu_id = 0 if self.__nr_gpus == 0 else tid % self.__nr_gpus
        print("[Manifold] Adding a worker with tid: ", tid)
        self.__workers.append(Worker(tid, gpu_id, f))
        self.__worker_lock.release()

    def add_idmap(self, ident, tid):
        self.__idmap_lock.acquire()
        self.__idmap[ident] = tid
        self.__idmap_lock.release()

    def join_all(self):
        for worker in self.__workers:
            worker.join()
        print("[Manifold] All workers joined")

    def barrier(self, stream_sync=False):
        if stream_sync:
            stream = torch.cuda.current_stream()
            stream.synchronize()
        self.__barrier.wait()

    def get_tid_from_idmap(self, ident):
        self.__idmap_lock.acquire()
        tid = self.__idmap[ident]
        self.__idmap_lock.release()
        return tid

    def get_worker_from_workers(self, tid):
        self.__worker_lock.acquire()
        worker = self.__workers[tid]
        self.__worker_lock.release()
        return worker

def GetWorker(tid):
    return Controller().get_worker_from_workers(tid)

def GetCurrentWorker():
    ident = threading.get_native_id()
    tid = Controller().get_tid_from_idmap(ident)
    return GetWorker(tid)

# For C++
def cuda_send_plugin(peer, src, src_size):
    my_worker = GetCurrentWorker()
    stream = torch.cuda.current_stream()
    stream.synchronize()
    my_worker.send_async(peer, src, src_size, stream)
    Controller().barrier()

def cuda_recv_plugin(recv_buf, recv_size):
    my_worker = GetCurrentWorker()
    my_worker.recv_async(recv_buf, recv_size)
    Controller().barrier()

def barrier_plugin():
    Controller().barrier()