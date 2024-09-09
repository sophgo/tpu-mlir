from tools.tdb import TdbCmdBackend
import time


def timeit(file):
    tdb = TdbCmdBackend(file, args={})
    tdb._reset()

    start = time.time()
    for i, cmd in enumerate(tdb.cmditer):
        cmd.reg.cmd_id_dep = i

    end = time.time()

    print((end - start) / i, i, end - start)
