from datetime import datetime as dt


class Timer:
    def __init__(self):
        self.start_time = dt.now()

    def reset(self):
        self.start_time = dt.now()

    def elapsed(self):
        end_time = dt.now()
        elapsed_time = end_time - self.start_time
        return elapsed_time
    
    def __str__(self):
        elapsed_time = self.elapsed()
        return "%02d - %02d:%02d:%02d" % (elapsed_time.days, elapsed_time.seconds // 3600, elapsed_time.seconds // 60 % 60, elapsed_time.seconds % 60)
