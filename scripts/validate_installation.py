#!/usr/bin/env python3
"""
Installation validation script for ftmwpipeline.

This script checks that the package is properly installed and all
dependencies are working correctly.
"""

import sys
from pathlib import Path

# Add source directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Run installation validation."""
    print("=== ftmwpipeline Installation Validation ===\n")

    # Test basic import
    print("1. Testing basic import...")
    try:
        import ftmwpipeline as fmw

        print(f"   ✅ Successfully imported ftmwpipeline {fmw.__version__}")
    except ImportError as e:
        print(f"   ❌ Failed to import ftmwpipeline: {e}")
        return 1

    # Test all submodules
    print("\n2. Testing submodule imports...")
    submodules = [
        "core",
        "preprocessing",
        "peak_detection",
        "window_assignment",
        "fitting",
        "visualization",
        "io",
        "config",
        "utils",
    ]

    failed_modules = []
    for module_name in submodules:
        try:
            module = getattr(fmw, module_name)
            print(f"   ✅ {module_name}: {len(module.__all__)} items")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")
            failed_modules.append(module_name)

    # Test pipeline creation
    print("\n3. Testing Pipeline creation...")
    try:
        pipeline = fmw.Pipeline()
        summary = pipeline.get_summary()
        print(f"   ✅ Pipeline created successfully")
        print(f"   Summary: {summary}")
    except Exception as e:
        print(f"   ❌ Failed to create Pipeline: {e}")
        return 1

    # Test workflow functions
    print("\n4. Testing workflow functions...")
    try:
        validation_results = fmw.workflows.validate_installation()
        print("   Installation validation results:")
        for component, status in validation_results.items():
            status_str = "✅ OK" if status else "❌ FAILED"
            print(f"     {component}: {status_str}")
    except Exception as e:
        print(f"   ❌ Failed to run validation: {e}")
        return 1

    # Test dependencies
    print("\n5. Testing dependencies...")
    dependencies = [
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pd"),
        ("h5py", "h5py"),
        ("yaml", "yaml"),
        ("tqdm", "tqdm"),
    ]

    failed_deps = []
    for dep_name, import_name in dependencies:
        try:
            if import_name == "yaml":
                import yaml
            elif import_name == "pd":
                import pandas as pd
            elif import_name == "np":
                import numpy as np
            else:
                __import__(import_name)
            print(f"   ✅ {dep_name}")
        except ImportError:
            print(f"   ❌ {dep_name} (not found)")
            failed_deps.append(dep_name)

    # Test CLI
    print("\n6. Testing CLI interface...")
    try:
        from ftmwpipeline.cli import main as cli_main

        # Test help command
        result = cli_main(["--help"])
        print(f"   ✅ CLI interface accessible")
    except SystemExit:
        print(f"   ✅ CLI interface working (help command)")
    except Exception as e:
        print(f"   ❌ CLI interface failed: {e}")

    # Summary
    print("\n=== Validation Summary ===")
    if failed_modules:
        print(f"❌ Failed submodules: {', '.join(failed_modules)}")
    if failed_deps:
        print(f"⚠️  Missing dependencies: {', '.join(failed_deps)}")

    if not failed_modules and not failed_deps:
        print("✅ All tests passed! ftmwpipeline is ready to use.")
        print("\nNote: This is Phase 1 infrastructure. Algorithm implementation")
        print("will be added in subsequent development phases.")
        return 0
    else:
        print("❌ Some issues found. Check installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
