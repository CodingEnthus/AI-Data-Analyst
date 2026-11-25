"""
Standalone Test Runner for ML Engine Test Suite.
"""
import sys
import os
import time
import inspect

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import tests.test_ml_engine as t

def run_all_tests():
    test_functions = [
        getattr(t, name)
        for name in dir(t)
        if name.startswith("test_") and callable(getattr(t, name))
    ]

    print(f"\n==================== Running {len(test_functions)} ML Engine Tests ====================")
    passed = 0
    failed = 0
    start_total = time.time()

    for func in test_functions:
        func_name = func.__name__
        print(f"\n▶ Running: {func_name} ...", end=" ")
        t0 = time.time()
        try:
            func()
            elapsed = round(time.time() - t0, 3)
            print(f"PASSED ({elapsed}s)")
            passed += 1
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"FAILED ({elapsed}s)")
            print(f"   [Error]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = round(time.time() - start_total, 3)
    print(f"\n==================== Test Summary: {passed} Passed, {failed} Failed in {total_time}s ====================\n")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
