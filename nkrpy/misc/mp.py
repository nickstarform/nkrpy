"""Multi-process/threading wrapper."""

# internal modules
import subprocess

# external modules

# relative modules

# global attributes
__all__ = ('run_seq_command', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def run_seq_command(command):
    """Uniform sequential caller."""
    print('Command passed: ', command)
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')


# end of code


"""
import threading
import multiprocessing

class WorkerThread(threading.Thread):
    def __init__(self, queue, size):
        threading.Thread.__init__(self)
        self.queue = queue
        self.size = size

    def run(self):
        self.queue.put(range(size))


class WorkerProcess(multiprocessing.Process):
    def __init__(self, queue, size):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.size = size

    def run(self):
        self.queue.put(range(size))


if __name__ == "__main__":
    size = 100000
    queue = multiprocessing.Queue()

    worker_t = WorkerThread(queue, size)
    worker_p = WorkerProcess(queue, size)

    worker_t.start()
    worker_t.join()
    print 'thread results length:', len(queue.get())

    worker_p.start()
    worker_p.join()
    print 'process results length:', len(queue.get())
"""

# end of file
