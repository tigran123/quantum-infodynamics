import sys
from time import time

class ProgressBar():
    def __init__(self, nsteps, width=30, msg="Processing"):
        self._start = self._stop = time()
        self._nsteps = nsteps
        self._width = width
        self._status = ""
        self._msg = msg

    def update(self, step):
        self._start = self._stop
        self._stop = time()
        self._status = self._stop - self._start

        progress = float(step)/float(self._nsteps - 1)
        if progress >= 1:
            progress = 1
        block = int(round(self._width * progress))
        text = "\r" + self._msg + ": [{}] {:.1%} {:.3}".format("#" * block + "-" * (self._width - block), progress, self._status)
        sys.stdout.write(text)
        sys.stdout.flush()
