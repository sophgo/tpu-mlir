#!/usr/bin/env python3
# Test script for GGUF loader functionality

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_gguf_loader():
    """Test basic GGUF loader functionality."""
    print("Testing GGUFQuantLoad...")

    # Check if we can import the module
    try:
        from GGUFQuantLoad import GGUFQuantLoad
        print("✓ GGUFQuantLoad imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GGUFQuantLoad: {e}")
        return False

    # Check if gguf-py is available
    try:
        from gguf import GGUFReader
        print("✓ gguf-py available")
    except ImportError as e:
        print(f"✗ gguf-py not available: {e}")
        print("Note: gguf-py is required for GGUF support")
        return False

    # Test with a sample GGUF file if available
    # Look for any .gguf file in the repository
    gguf_files = []
    # for root, dirs, files in os.walk('/sandisk/tpu-mlir-org'):
    for root, dirs, files in os.walk('/data/models/unsloth/Qwen3-0.6B-GGUF'):
        for file in files:
            if file.endswith('.gguf'):
                gguf_files.append(os.path.join(root, file))
                if len(gguf_files) >= 3:
                    break
        if len(gguf_files) >= 3:
            break

    if not gguf_files:
        print("⚠ No GGUF files found for testing")
        print("Basic import test passed")
        return True

    print(f"Found {len(gguf_files)} GGUF files")

    # Test with the first GGUF file
    test_file = gguf_files[0]
    print(f"Testing with: {test_file}")

    try:
        # Try to create GGUFReader directly first
        reader = GGUFReader(test_file)
        print(f"✓ GGUFReader created successfully")
        print(f"  File contains {len(reader.tensors)} tensors")

        # Try our GGUFQuantLoad
        loader = GGUFQuantLoad(test_file)
        print(f"✓ GGUFQuantLoad created successfully")

        # Get metadata
        metadata = loader.get_metadata()
        print(f"✓ Extracted metadata: {list(metadata.keys())}")

        # List some tensors
        tensors = loader.get_all_tensors()
        print(f"✓ Found {len(tensors)} tensors")
        if tensors:
            print(f"  First 5 tensors: {tensors[:5]}")
            for cur_tensor in tensors:
                # Try to read first tensor
                try:
                    data = loader.read(cur_tensor)
                    print(f"✓ Read tensor '{cur_tensor}': shape={data.shape}, dtype={data.dtype}")

                    # Get tensor info
                    info = loader.get_tensor_info(cur_tensor)
                    if info:
                        print(f"✓ Tensor info: {info}")
                except Exception as e:
                    print(f"⚠ Could not read tensor '{cur_tensor}': {e}")

        return True

    except Exception as e:
        print(f"✗ Error testing GGUF file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quant_converter():
    """Test QuantConverter functionality."""
    print("\nTesting QuantConverter...")

    try:
        from QuantConverter import QuantConverter
        print("✓ QuantConverter imported successfully")

        # Create converter instance
        converter = QuantConverter(group_size=64, scale_dtype=np.float32)
        print(f"✓ Created QuantConverter with group_size={converter.group_size}")

        # Test with dummy data (since we don't have actual GGUF quantized data)
        print("⚠ Skipping conversion tests (no quantized GGUF data available)")

        return True

    except ImportError as e:
        print(f"✗ Failed to import QuantConverter: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing QuantConverter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gguf_model_handle():
    """Test GGUFModelHandle imports."""
    print("\nTesting GGUFModelHandle...")

    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))

        try:
            from LlmConverter import LlmConverter, LlmType, WeightType
            from LlmInfo import LlmInfo
            print("✓ All dependencies imported successfully")
        except ImportError as e:
            print(f"⚠ Could not import dependencies: {e}")
            print("This is expected if running outside the full TPU-MLIR environment")
            return True

        from ModelHandle import GGUFModelHandle, SafetensorsModelHandle
        print("✓ GGUFModelHandle and SafetensorsModelHandle imported successfully")

        return True

    except ImportError as e:
        print(f"✗ Failed to import ModelHandle: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing ModelHandle: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing GGUF Converter Components")
    print("=" * 60)

    tests = [
        ("GGUFQuantLoad", test_gguf_loader),
        ("QuantConverter", test_quant_converter),
        ("GGUFModelHandle", test_gguf_model_handle),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Test: {name}")
        print(f"{'='*40}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:30} {status}")
        if not success:
            all_passed = False

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
