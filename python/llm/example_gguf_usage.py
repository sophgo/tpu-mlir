#!/usr/bin/env python3
"""
Example usage of GGUF converter for QWen3 models.

This script demonstrates how to use the GGUF converter components.
Note: This is a demonstration and requires actual GGUF model files to run.
"""

import sys
import os

# Add necessary paths
sys.path.insert(0, os.path.dirname(__file__))


def demonstrate_gguf_loader():
    """Demonstrate GGUFQuantLoad usage."""
    print("=== GGUFQuantLoad Demonstration ===")

    # Example: Loading a GGUF file
    print("\n1. Creating GGUFQuantLoad instance:")
    print("   loader = GGUFQuantLoad('path/to/model.gguf')")

    print("\n2. Reading metadata:")
    print("   metadata = loader.get_metadata()")
    print("   print(f'Model architecture: {metadata.get(\"architecture\")}')")
    print("   print(f'Hidden size: {metadata.get(\"hidden_size\")}')")

    print("\n3. Checking if tensor exists:")
    print("   exists = loader.is_exist('model.layers.0.self_attn.q_proj.weight')")

    print("\n4. Reading tensor (dequantizes if needed):")
    print("   data = loader.read('model.layers.0.self_attn.q_proj.weight')")
    print("   print(f'Tensor shape: {data.shape}, dtype: {data.dtype}')")

    print("\n5. Getting quantization info:")
    print("   info = loader.get_tensor_info('model.layers.0.self_attn.q_proj.weight')")
    print("   if info and info['is_quantized']:")
    print("       print(f'Quantization type: {info[\"quant_type\"]}')")


def demonstrate_quant_converter():
    """Demonstrate QuantConverter usage."""
    print("\n=== QuantConverter Demonstration ===")

    print("\n1. Creating QuantConverter instance:")
    print("   converter = QuantConverter(group_size=64, scale_dtype=np.float32)")

    print("\n2. Converting GGUF quantized tensor:")
    print("   # Assuming we have GGUF data and info")
    print("   converted = converter.convert_from_gguf(")
    print("       gguf_data,  # Raw GGUF tensor data")
    print("       quant_type, # GGMLQuantizationType")
    print("       original_shape # Original tensor shape")
    print("   )")
    print("   ")
    print("   print(f'qweight shape: {converted[\"qweight\"].shape}')")
    print("   print(f'scales shape: {converted[\"scales\"].shape}')")
    print("   print(f'bits: {converted[\"bits\"]}')")

    print("\n3. High-level conversion:")
    print("   result = converter.convert_to_llmconv_format(")
    print("       loader,    # GGUFQuantLoad instance")
    print("       'model.layers.0.self_attn.q_proj.weight',")
    print("       transpose=True")
    print("   )")


def demonstrate_gguf_model_handle():
    """Demonstrate GGUFModelHandle usage."""
    print("\n=== GGUFModelHandle Demonstration ===")

    print("\n1. Command line usage:")
    print("   python3 -m tools.llm_convert \\")
    print("     -m /path/to/qwen3-model.gguf \\")
    print("     -s 4096 \\")
    print("     -q w4bf16 \\")
    print("     -c bm1684x \\")
    print("     -o ./output")

    print("\n2. Key features:")
    print("   - Automatically detects GGUF file format")
    print("   - Preserves 4-bit/8-bit quantization")
    print("   - Converts GGUF block quantization to LlmConverter group quantization")
    print("   - Extracts model config from GGUF metadata")
    print("   - Generates MLIR with quantized operations")
    print("   - Works with any architecture (Qwen3, Llama, etc.) via LlmConverter")


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n=== Dependency Check ===")

    deps = [
        ("numpy", "Numerical computations"),
        ("gguf-py", "GGUF file reading"),
    ]

    for dep, desc in deps:
        try:
            if dep == "gguf-py":
                # Special handling for gguf-py
                sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                                '../../llama.cpp/gguf-py'))
                import gguf
                print(f"✓ {dep}: {desc}")
            else:
                __import__(dep)
                print(f"✓ {dep}: {desc}")
        except ImportError:
            print(f"✗ {dep}: {desc} (missing)")


def main():
    """Main demonstration."""
    print("=" * 70)
    print("GGUF Converter for QWen3 Models - Usage Demonstration")
    print("=" * 70)

    # Check dependencies
    check_dependencies()

    # Demonstrate components
    demonstrate_gguf_loader()
    demonstrate_quant_converter()
    demonstrate_gguf_model_handle()

    print("\n" + "=" * 70)
    print("Summary:")
    print("-" * 70)
    print("1. GGUFQuantLoad: Loads GGUF files with quantization preservation")
    print("2. QuantConverter: Converts GGUF quantization to LlmConverter format")
    print("3. GGUFModelHandle: Format-specific loader, composable with any converter")
    print("4. Updated llm_convert.py: Automatically detects GGUF files")
    print("\nTo convert a QWen3 GGUF model:")
    print("  python3 tools/llm_convert.py -m model.gguf -s 4096 -q w4bf16 -c bm1684x")
    print("=" * 70)


if __name__ == "__main__":
    main()
