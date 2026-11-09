import runpy, sys  # moved to lessonkit/runner.py
if __name__ == "__main__": runpy.run_module("lessonkit.runner", run_name="__main__")
else: from lessonkit import runner as _m; sys.modules[__name__] = _m
