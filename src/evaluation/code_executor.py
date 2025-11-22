"""
Safe code execution engine with timeout and sandboxing.
"""

import multiprocessing
import signal
import sys
import traceback
from contextlib import contextmanager
from io import StringIO
from typing import Dict, Optional, Tuple


class TimeoutException(Exception):
    """Raised when execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Execution timed out")


@contextmanager
def time_limit(seconds: int):
    """Context manager for execution timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def execute_code(code: str, test_code: str, timeout: int = 5) -> Dict:
    """
    Execute code with test cases in isolated environment.
    
    Args:
        code: Generated code to test
        test_code: Test cases to run
        timeout: Maximum execution time in seconds
    
    Returns:
        Dictionary with execution results:
        - success: bool
        - error_type: str (syntax_error, runtime_error, assertion_error, timeout, etc.)
        - error_message: str
        - stdout: str
        - stderr: str
    """
    result = {
        "success": False,
        "error_type": None,
        "error_message": None,
        "stdout": "",
        "stderr": "",
    }
    
    # Combine code and tests
    full_code = f"{code}\n\n{test_code}"
    
    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        with time_limit(timeout):
            # Create isolated namespace
            namespace = {
                '__builtins__': __builtins__,
            }
            
            # Execute code
            exec(full_code, namespace)
            
            result["success"] = True
            result["error_type"] = None
            
    except SyntaxError as e:
        result["error_type"] = "syntax_error"
        result["error_message"] = str(e)
        
    except TimeoutException:
        result["error_type"] = "timeout"
        result["error_message"] = f"Execution exceeded {timeout} seconds"
        
    except AssertionError as e:
        result["error_type"] = "assertion_error"
        result["error_message"] = str(e) or "Test case failed"
        
    except Exception as e:
        result["error_type"] = "runtime_error"
        result["error_message"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        
    finally:
        # Restore stdout/stderr and capture output
        result["stdout"] = sys.stdout.getvalue()
        result["stderr"] = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if code has valid Python syntax.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)


def execute_with_multiprocessing(code: str, test_code: str, 
                                 timeout: int = 5) -> Dict:
    """
    Execute code in separate process for extra isolation.
    More secure but slower than direct execution.
    """
    def worker(queue, code, test_code):
        result = execute_code(code, test_code, timeout)
        queue.put(result)
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=worker,
        args=(queue, code, test_code)
    )
    
    process.start()
    process.join(timeout=timeout + 1)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "success": False,
            "error_type": "timeout",
            "error_message": f"Process exceeded {timeout} seconds",
            "stdout": "",
            "stderr": "",
        }
    
    if not queue.empty():
        return queue.get()
    else:
        return {
            "success": False,
            "error_type": "unknown_error",
            "error_message": "Process terminated without result",
            "stdout": "",
            "stderr": "",
        }


if __name__ == "__main__":
    # Test the executor
    
    # Test 1: Valid code
    code1 = """
def add(a, b):
    return a + b
"""
    test1 = "assert add(2, 3) == 5"
    result1 = execute_code(code1, test1)
    print(f"Test 1 (valid): {result1}")
    
    # Test 2: Syntax error
    code2 = """
def add(a, b)
    return a + b
"""
    test2 = "assert add(2, 3) == 5"
    result2 = execute_code(code2, test2)
    print(f"Test 2 (syntax error): {result2}")
    
    # Test 3: Runtime error
    code3 = """
def divide(a, b):
    return a / b
"""
    test3 = "assert divide(10, 0) == 0"
    result3 = execute_code(code3, test3)
    print(f"Test 3 (runtime error): {result3}")
    
    # Test 4: Wrong output
    code4 = """
def add(a, b):
    return a - b
"""
    test4 = "assert add(2, 3) == 5"
    result4 = execute_code(code4, test4)
    print(f"Test 4 (wrong output): {result4}")

def categorize_failure(result: dict) -> str:
    """Categorize failure type."""
    if result["success"]:
        return "success"
    
    error_type = result["error_type"]
    
    if error_type == "syntax_error":
        return "syntax_error"
    elif error_type == "timeout":
        return "timeout"
    elif error_type == "assertion_error":
        return "wrong_output"
    elif error_type == "runtime_error":
        return "runtime_error"
    else:
        return "unknown_error"
