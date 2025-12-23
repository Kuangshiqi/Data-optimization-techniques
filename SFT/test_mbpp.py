import multiprocessing
from typing import List, Optional
import traceback

def _eval_worker(completion: str, tests: List[str], test_imports: List[str], result_queue: multiprocessing.Queue):
    globals_dict = {}
    try:
        for import_stmt in test_imports:
            exec(import_stmt, globals_dict)
        exec(completion, globals_dict)

        passed_count = 0
        for test in tests:
            try:
                exec(test, globals_dict)
                passed_count += 1
            except Exception:
                # 只记录失败，不中断后续测试
                # （可选：把单个测试的 traceback 放入一个列表以便调试）
                pass

        # 最终放回通过数（整数）
        result_queue.put(passed_count)

    except Exception:
        # 如果导入或补全本身就出错，把 traceback 放回去以便调试
        result_queue.put(("ERROR", traceback.format_exc()))

def eval_code(completion: str,
              tests: List[str],
              test_imports: Optional[List[str]] = None,
              timeout: float = 2.0) -> float:
    if test_imports is None:
        test_imports = []

    total = len(tests)
    if total == 0:
        return 0.0

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_eval_worker,
        args=(completion, tests, test_imports, result_queue)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return 0.0

    # 尝试从队列安全拿到结果
    try:
        result = result_queue.get(timeout=0.1)
    except Exception:
        return 0.0

    # 处理两种可能的返回：整数 or 错误信息
    if isinstance(result, int):
        return result / total
    if isinstance(result, tuple) and result and result[0] == "ERROR":
        # 可以打印或记录 result[1]（traceback）供调试
        return 0.0
    # 其它情况视为失败
    return 0.0
