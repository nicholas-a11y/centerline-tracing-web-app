import runpy, sys  # moved to lessonkit/trace_template.py
if __name__ == "__main__": runpy.run_module("lessonkit.trace_template", run_name="__main__")
else: from lessonkit import trace_template as _m; sys.modules[__name__] = _m
