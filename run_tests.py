"""
Unified test runner - Run this before training
"""

import sys
import argparse


def run_quick_check():
    """Quick sanity check (1-2 minutes)"""
    print("\n" + "="*60)
    print("QUICK SANITY CHECK")
    print("="*60)
    print("This will:")
    print("  1. Check GPU availability")
    print("  2. Test data format")
    print("  3. Run a few training steps")
    print("="*60 + "\n")

    try:
        from test_data_format import main as test_format
        result = test_format()
        if result != 0:
            return 1

        print("\n✓ Quick check PASSED! You're ready to train.\n")
        return 0

    except Exception as e:
        print(f"\n✗ Quick check FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


def run_full_test():
    """Full test with more data"""
    print("\n" + "="*60)
    print("FULL SYSTEM TEST")
    print("="*60)
    print("This will test with more data (may take 5-10 minutes)")
    print("="*60 + "\n")

    try:
        from quick_test import quick_test
        quick_test()
        print("\n✓ Full test PASSED!\n")
        return 0

    except Exception as e:
        print(f"\n✗ Full test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


def check_gpu_only():
    """Just check GPU"""
    from check_gpu import check_gpu
    check_gpu()


def main():
    parser = argparse.ArgumentParser(description='Run tests before training')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'gpu'],
                       help='Test mode: quick (fast), full (thorough), gpu (GPU only)')

    args = parser.parse_args()

    if args.mode == 'quick':
        return run_quick_check()
    elif args.mode == 'full':
        return run_full_test()
    elif args.mode == 'gpu':
        check_gpu_only()
        return 0


if __name__ == "__main__":
    sys.exit(main())
