import re
import sys
import numpy as np
import multiprocessing
import testing_util as test_util

sys.set_int_max_str_digits(0)


def check_correctness(in_outs, code, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(in_outs, code, debug, result):
        try:
            if debug:
                print(f"Running test for problem: {in_outs}")
            result.append(test_util.run_test(in_outs, code, debug))
            if debug:
                print(f"Test completed with result: {result}")
        except Exception as e:
            if debug:
                print(f"Error in _temp_run: {e}")

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, code, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            print(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"Global timeout occurred, returning default result.")
    if debug:
        print(f"Final result: {result}")
    return result[0]


def eval_code(in_outs, code, TIMEOUT=10):
    # print(len(in_outs['inputs']), len(in_outs['outputs']))
    res = [-2]
    try:
        res = check_correctness(in_outs, code, timeout=TIMEOUT, debug=False)
        fixed = []
        for e in res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        res = fixed
    except Exception as e:
        print(f"test framework exception = {repr(e)}{e}\n")
    finally:
        assert isinstance(res, list)

    pass_ratio = res.count(True) / len(in_outs['inputs'])

    return pass_ratio
