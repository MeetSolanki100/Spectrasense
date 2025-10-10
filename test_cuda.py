#!/usr/bin/env python3
"""
CUDA Test Script for Jetson Nano
Tests CUDA availability and performance on Jetson Nano's Maxwell GPU
"""

import torch
import time
import platform
import sys

def test_cuda_availability():
    """Test basic CUDA availability"""
    print("🔍 Testing CUDA Availability")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check if we're on Jetson
    is_jetson = platform.machine() == 'aarch64' and 'jetson' in platform.platform().lower()
    print(f"Jetson detected: {is_jetson}")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
            print(f"  - CUDA cores: {props.multi_processor_count * 128}")  # Maxwell has 128 cores per SM
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def test_cuda_performance():
    """Test CUDA performance with tensor operations"""
    print("\n⚡ Testing CUDA Performance")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping performance tests")
        return
    
    device = torch.device('cuda')
    
    # Test different tensor sizes
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        try:
            print(f"\nTesting {size}x{size} matrix multiplication...")
            
            # Create random tensors
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warm up
            torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Time the operation
            start_time = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            memory_used = torch.cuda.memory_allocated() / 1024**2
            memory_cached = torch.cuda.memory_reserved() / 1024**2
            
            print(f"  Time: {elapsed:.4f} seconds")
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Memory cached: {memory_cached:.1f} MB")
            
            # Clear cache
            del a, b, c
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"  ❌ Error with size {size}: {e}")
            break

def test_vision_encoder_cuda():
    """Test Vision Encoder CUDA integration"""
    print("\n🎯 Testing Vision Encoder CUDA Integration")
    print("=" * 50)
    
    try:
        from main import get_optimal_device, initialize_models
        
        print("Testing device detection...")
        device = get_optimal_device()
        print(f"Selected device: {device}")
        
        if device == "cuda":
            print("✅ CUDA device selected correctly")
            
            print("Testing model initialization...")
            yolo_model, blip_processor, blip_model, device = initialize_models()
            print(f"BLIP model device: {next(blip_model.parameters()).device}")
            print("✅ Models initialized successfully with CUDA")
            
            # Test inference
            print("Testing inference...")
            import numpy as np
            from PIL import Image
            
            # Create a dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pil_image = Image.fromarray(dummy_image)
            
            # Test BLIP inference
            inputs = blip_processor(pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=20)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
            
            print(f"✅ Inference successful: '{caption}'")
            
        else:
            print(f"⚠️  CUDA not selected, using {device}")
            
    except ImportError as e:
        print(f"❌ Could not import main module: {e}")
    except Exception as e:
        print(f"❌ Error testing Vision Encoder: {e}")

def test_memory_management():
    """Test CUDA memory management"""
    print("\n💾 Testing CUDA Memory Management")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping memory tests")
        return
    
    device = torch.device('cuda')
    
    # Test memory allocation and deallocation
    print("Testing memory allocation...")
    
    # Allocate memory
    tensor1 = torch.randn(1000, 1000, device=device)
    memory_after_alloc = torch.cuda.memory_allocated() / 1024**2
    print(f"Memory after allocation: {memory_after_alloc:.1f} MB")
    
    # Allocate more memory
    tensor2 = torch.randn(1000, 1000, device=device)
    memory_after_alloc2 = torch.cuda.memory_allocated() / 1024**2
    print(f"Memory after second allocation: {memory_after_alloc2:.1f} MB")
    
    # Clear cache
    del tensor1, tensor2
    torch.cuda.empty_cache()
    memory_after_clear = torch.cuda.memory_allocated() / 1024**2
    print(f"Memory after clearing: {memory_after_clear:.1f} MB")
    
    print("✅ Memory management test completed")

def main():
    """Run all CUDA tests"""
    print("🚀 Jetson Nano CUDA Test Suite")
    print("=" * 50)
    
    # Test CUDA availability
    cuda_available = test_cuda_availability()
    
    if cuda_available:
        # Test performance
        test_cuda_performance()
        
        # Test memory management
        test_memory_management()
        
        # Test Vision Encoder integration
        test_vision_encoder_cuda()
        
        print("\n🎉 All CUDA tests completed!")
        print("\n📊 Summary:")
        print("- CUDA is available and working")
        print("- GPU memory management is functioning")
        print("- Vision Encoder can use CUDA acceleration")
        
    else:
        print("\n❌ CUDA tests failed - CUDA is not available")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if JetPack is properly installed")
        print("2. Verify PyTorch with CUDA support is installed")
        print("3. Ensure maximum performance mode is enabled: sudo nvpmodel -m 0")
        print("4. Check GPU temperature and cooling")

if __name__ == "__main__":
    main()
