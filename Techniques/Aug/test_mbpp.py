import threading
from typing import List, Optional

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout: float = 2.0):
    if kwargs is None:
        kwargs = {}
    result = {'value': None, 'error': None}

    def target():
        try:
            result['value'] = func(*args, **kwargs)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException('Timeout')
    if result['error'] is not None:
        raise result['error']
    return result['value']


def eval_code(completion: str,
                  tests: List[str],
                  test_imports: Optional[List[str]] = None,
                  timeout: float = 2.0) -> float:
    if test_imports is None:
        test_imports = []

    total = len(tests)
    if total == 0:
        return 0.0

    passed_count = 0

    for i, test in enumerate(tests, start=1):
        globals_dict = {}
        try:
            for import_stmt in test_imports:
                run_with_timeout(exec, (import_stmt, globals_dict), {}, timeout)

            run_with_timeout(exec, (completion, globals_dict), {}, timeout)
            run_with_timeout(exec, (test, globals_dict), {}, timeout)
            passed_count += 1

        except TimeoutException:
            continue
        except Exception:
            continue

    return passed_count / total
