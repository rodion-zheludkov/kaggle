from collections import Iterable
from threading import Thread
import sys

from itertools import izip_longest
from multiprocessing import Process, Pipe
from itertools import izip


class FilesIO(Iterable):
    def __iter__(self):
        return self

    def __init__(self, files, *args, **kwargs):
        self.files = files
        self.active = 0

    def readline(self, *args, **kwargs):
        ll = self.files[self.active].readline(*args, **kwargs)
        #print ll
        if not ll:
            if self.active == (len(self.files) - 1):
                return ''
            else:
                self.active += 1
                return self.readline()
        else:
            return ll

    def next(self):
        ll = self.readline()
        if not ll:
            raise StopIteration
        return ll


def run_log_thread(process):
    def thread_log():
        while True:
            nextline = process.stdout.readline()
            if nextline == '' and process.poll() is not None:
                break
            sys.stdout.write(nextline)
            sys.stdout.flush()

    t = Thread(target=thread_log)
    t.daemon = True
    t.start()


def spawn(f):
    def fun(pipe, x):
        pipe.send(f(x))
        pipe.close()

    return fun


def parmap(f, xx, N = 4):
    # xx = range(xx)
    batches = izip_longest(*(iter(xx),) * N)

    result = []
    for batch in batches:
        print 'Doing batch ', batch
        batch = filter(lambda x: x is not None, batch)
        pipe = [Pipe() for x in batch]
        proc = [Process(target=spawn(f), args=(c, x)) for x, (p, c) in izip(batch, pipe)]
        [p.start() for p in proc]
        [p.join() for p in proc]
        result += [p.recv() for (p, c) in pipe]

    return result


if __name__ == '__main__':
    with open('1.txt', 'r') as f1, open('2.txt', 'r') as f2:
        ff = FilesIO([f1, f2])
        for line in ff:
            print '>>' + line.strip()