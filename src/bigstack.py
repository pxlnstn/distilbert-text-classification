"""Run a script inside a thread with a bigger stack.

On Windows + Python 3.13 the torch/transformers/sklearn/pandas import chain can
overflow the default ~8 MB thread stack and crash. A 64 MB stack fixes it. The heavy
imports have to go inside the function passed to run() so they run on this thread.
"""

import threading

STACK_BYTES = 64 * 1024 * 1024


def run(main):
    threading.stack_size(STACK_BYTES)
    box: dict = {}

    def target():
        try:
            box["value"] = main()
        except BaseException as exc:  # re-raise on the main thread
            box["error"] = exc

    t = threading.Thread(target=target, name="bigstack-main")
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")
