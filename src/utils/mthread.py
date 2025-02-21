import threading
from functools import wraps

def thread_parallel(func):
    """
    Start a new thread to run the function. The return value of the function is ignored.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建线程
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        # 启动线程
        thread.start()
        # 返回线程对象，以便外部可以进行线程相关的操作，如join等
        return thread
    return wrapper