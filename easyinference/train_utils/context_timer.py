from time import time


class ContextTimer:
    """ Usage:

    with ContextTimer() as timer:
        #do stuff
        print("Current time: ", timer.time)
        # do more stuff
        print("Current time: ", timer.time
    print("Total process took : ", timer.time, " seconds")
        """

    def __init__(self, name="GenericTimer", post_print=True):
        self.name = name
        self.post_print = post_print

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time()
        if self.post_print:
            final_elapsed = self.end_time - self.start_time
            fps = None if final_elapsed == 0 else 1 / final_elapsed
            p = self.name + ": " + str(final_elapsed)
            if fps is not None:
                p += " FPS: " + str(fps)
            print(p)

    @property
    def elapsed(self):
        return time() - self.start_time
