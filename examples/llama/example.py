#from tensorrt_llm.libs import manifoldwrapper as manifold
#from tensorrt_llm.manifold import Controller, Worker
import tensorrt_llm

def foo():
    worker = tensorrt_llm.GetCurrentWorker()
    print("hello from foo function: ", worker.get_tid())
    pass

def main():
    #controller = manifold.ControllerWrapper()
    ##print("create worker from now")
    #controller.add_worker(0, foo)
    #controller.add_worker(1, foo)
    #controller.add_worker(2, foo)
    #controller.add_worker(3, foo)
    #controller.join_all()

    controller = tensorrt_llm.Controller()
    controller2 = tensorrt_llm.Controller()
    controller.add_worker(0, foo)
    controller.add_worker(1, foo)
    controller.add_worker(2, foo)
    controller2.add_worker(3, foo)
    controller.join_all()


if __name__ == '__main__':
    main()
