#!/usr/bin/env python3
"""
TOPS (Tera Operations Per Second) Benchmark
Measures AI inference performance in TOPS across different hardware.
"""

import time
import json
import argparse
import numpy as np
import platform
import psutil
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import pynvml
except ImportError:
    pynvml = None

class TOPSBenchmark:
    def __init__(self):
        self.device = self._get_best_device()
        self.system_info = self._collect_system_info()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'system_info': self.system_info,
            'tops_measurements': {}
        }
    
    def _collect_system_info(self):
        """Collect detailed system information."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
        }
        
        # CPU Information
        cpu_info = {
            'logical_cores': psutil.cpu_count(logical=True),
            'physical_cores': psutil.cpu_count(logical=False),
        }
        
        # Get CPU frequency if available
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['max_frequency_mhz'] = freq.max
                cpu_info['current_frequency_mhz'] = freq.current
        except:
            pass
        
        # Get detailed CPU info if cpuinfo is available
        if cpuinfo:
            try:
                detailed_cpu = cpuinfo.get_cpu_info()
                cpu_info['brand'] = detailed_cpu.get('brand_raw', 'Unknown')
                cpu_info['arch'] = detailed_cpu.get('arch', 'Unknown')
                cpu_info['vendor'] = detailed_cpu.get('vendor_id_raw', 'Unknown')
            except:
                cpu_info['brand'] = platform.processor()
        else:
            cpu_info['brand'] = platform.processor()
        
        info['cpu'] = cpu_info
        
        # Memory Information
        memory = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2)
        }
        
        # GPU Information
        gpu_info = {'detected_gpus': []}
        
        # Try to get NVIDIA GPU info
        if pynvml:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info['detected_gpus'].append({
                        'index': i,
                        'name': name,
                        'memory_gb': round(memory_info.total / (1024**3), 2),
                        'type': 'NVIDIA'
                    })
            except:
                pass
        
        # PyTorch GPU info
        if torch:
            if torch.cuda.is_available():
                gpu_info['pytorch_cuda'] = {
                    'available': True,
                    'device_count': torch.cuda.device_count(),
                    'cuda_version': torch.version.cuda,
                    'current_device': torch.cuda.current_device()
                }
                
                # Get current device properties
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    gpu_info['pytorch_cuda']['device_name'] = props.name
                    gpu_info['pytorch_cuda']['total_memory_gb'] = round(props.total_memory / (1024**3), 2)
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['pytorch_mps'] = {
                    'available': True,
                    'device_type': 'Apple Silicon'
                }
            else:
                gpu_info['pytorch_cuda'] = {'available': False}
        
        info['gpu'] = gpu_info
        
        return info
    
    def _print_system_info(self):
        """Print system information in a nice format."""
        print("💻 SYSTEM INFORMATION")
        print("=" * 50)
        
        # Platform
        print(f"Platform: {self.system_info['platform']}")
        
        # CPU
        cpu = self.system_info['cpu']
        print(f"CPU: {cpu.get('brand', 'Unknown')}")
        print(f"Cores: {cpu['physical_cores']} physical, {cpu['logical_cores']} logical")
        if 'max_frequency_mhz' in cpu:
            print(f"Frequency: {cpu['max_frequency_mhz']:.0f} MHz (max)")
        
        # Memory
        memory = self.system_info['memory']
        print(f"RAM: {memory['total_gb']} GB total, {memory['available_gb']} GB available")
        
        # GPU
        gpu = self.system_info['gpu']
        if gpu['detected_gpus']:
            print("GPU(s):")
            for gpu_device in gpu['detected_gpus']:
                print(f"  - {gpu_device['name']} ({gpu_device['memory_gb']} GB)")
        
        if torch and gpu.get('pytorch_cuda', {}).get('available'):
            cuda_info = gpu['pytorch_cuda']
            print(f"CUDA: {cuda_info['cuda_version']} (PyTorch)")
        elif torch and gpu.get('pytorch_mps', {}).get('available'):
            print("Apple MPS: Available (PyTorch)")
        else:
            print("GPU Acceleration: Not available (CPU only)")
        
        print(f"Compute Device: {self.device}")
        print()
    
    def _get_best_device(self):
        """Get the best available device."""
        if torch and torch.cuda.is_available():
            return torch.device('cuda')
        elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _count_conv2d_ops(self, input_shape, kernel_shape, stride=1, padding=0):
        """Count operations in a Conv2D layer."""
        batch_size, in_channels, in_height, in_width = input_shape
        out_channels, _, kernel_height, kernel_width = kernel_shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1
        
        # Each output pixel requires kernel_height * kernel_width * in_channels multiply-adds
        ops_per_output = kernel_height * kernel_width * in_channels
        total_outputs = batch_size * out_channels * out_height * out_width
        
        return ops_per_output * total_outputs
    
    def _count_linear_ops(self, input_size, output_size, batch_size):
        """Count operations in a Linear layer."""
        return batch_size * input_size * output_size
    
    def measure_conv2d_tops(self, batch_size=32, runs=50):
        """Measure TOPS for Conv2D operations."""
        if not torch:
            return None
        
        print(f"Measuring Conv2D TOPS on {self.device}...")
        
        # Define conv layer
        conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(self.device)
        input_tensor = torch.randn(batch_size, 64, 224, 224, device=self.device)
        
        # Count operations
        input_shape = input_tensor.shape
        kernel_shape = (128, 64, 3, 3)
        total_ops = self._count_conv2d_ops(input_shape, kernel_shape, stride=1, padding=1)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = conv(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start_time = time.time()
                output = conv(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        ops_per_second = total_ops / avg_time
        tops = ops_per_second / 1e12
        
        return {
            'operation': 'Conv2D',
            'batch_size': batch_size,
            'input_shape': list(input_shape),
            'total_ops': total_ops,
            'avg_time_seconds': avg_time,
            'ops_per_second': ops_per_second,
            'tops': tops,
            'runs': runs
        }
    
    def measure_linear_tops(self, batch_size=64, runs=100):
        """Measure TOPS for Linear (Dense) operations."""
        if not torch:
            return None
        
        print(f"Measuring Linear TOPS on {self.device}...")
        
        # Define linear layer
        linear = nn.Linear(2048, 1000).to(self.device)
        input_tensor = torch.randn(batch_size, 2048, device=self.device)
        
        # Count operations
        total_ops = self._count_linear_ops(2048, 1000, batch_size)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = linear(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start_time = time.time()
                output = linear(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        ops_per_second = total_ops / avg_time
        tops = ops_per_second / 1e12
        
        return {
            'operation': 'Linear',
            'batch_size': batch_size,
            'input_size': 2048,
            'output_size': 1000,
            'total_ops': total_ops,
            'avg_time_seconds': avg_time,
            'ops_per_second': ops_per_second,
            'tops': tops,
            'runs': runs
        }
    
    def measure_matrix_multiply_tops(self, size=2048, runs=50):
        """Measure TOPS for matrix multiplication."""
        if not torch:
            return None
        
        print(f"Measuring Matrix Multiply TOPS on {self.device}...")
        
        # Create matrices
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        # Count operations (2 * n^3 for n x n matrix multiply)
        total_ops = 2 * size ** 3
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = torch.mm(a, b)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start_time = time.time()
                result = torch.mm(a, b)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        ops_per_second = total_ops / avg_time
        tops = ops_per_second / 1e12
        
        return {
            'operation': 'Matrix Multiply',
            'matrix_size': size,
            'total_ops': total_ops,
            'avg_time_seconds': avg_time,
            'ops_per_second': ops_per_second,
            'tops': tops,
            'runs': runs
        }
    
    def measure_resnet_inference_tops(self, batch_size=16, runs=30):
        """Measure TOPS for ResNet inference."""
        if not torch:
            return None
        
        print(f"Measuring ResNet Inference TOPS on {self.device}...")
        
        try:
            import torchvision.models as models
            
            # Load ResNet-50
            model = models.resnet50(weights=None).to(self.device)
            model.eval()
            
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            # Estimate operations for ResNet-50 (approximately 4.1 GMACs per image)
            gmacs_per_image = 4.1
            total_ops = batch_size * gmacs_per_image * 1e9  # Convert to operations
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(runs):
                    start_time = time.time()
                    output = model(input_tensor)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            ops_per_second = total_ops / avg_time
            tops = ops_per_second / 1e12
            
            return {
                'operation': 'ResNet-50 Inference',
                'batch_size': batch_size,
                'estimated_ops': total_ops,
                'avg_time_seconds': avg_time,
                'ops_per_second': ops_per_second,
                'tops': tops,
                'runs': runs
            }
            
        except ImportError:
            print("TorchVision not available, skipping ResNet benchmark")
            return None
    
    def measure_numpy_tops(self, size=2000, runs=20):
        """Measure TOPS for NumPy operations (CPU baseline)."""
        print("Measuring NumPy TOPS (CPU)...")
        
        # Create matrices
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Count operations
        total_ops = 2 * size ** 3
        
        # Warmup
        for _ in range(3):
            _ = np.dot(a, b)
        
        # Benchmark
        times = []
        for _ in range(runs):
            start_time = time.time()
            result = np.dot(a, b)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        ops_per_second = total_ops / avg_time
        tops = ops_per_second / 1e12
        
        return {
            'operation': 'NumPy Matrix Multiply',
            'matrix_size': size,
            'total_ops': total_ops,
            'avg_time_seconds': avg_time,
            'ops_per_second': ops_per_second,
            'tops': tops,
            'runs': runs
        }
    
    def run_all_tops_benchmarks(self):
        """Run all TOPS benchmarks."""
        # Print system information first
        self._print_system_info()
        
        print("🚀 RUNNING TOPS BENCHMARKS")
        print("=" * 50)
        
        # NumPy baseline (always available)
        numpy_result = self.measure_numpy_tops()
        if numpy_result:
            self.results['tops_measurements']['numpy'] = numpy_result
            print(f"NumPy TOPS: {numpy_result['tops']:.2f}")
        
        if torch:
            # Matrix multiply
            matmul_result = self.measure_matrix_multiply_tops()
            if matmul_result:
                self.results['tops_measurements']['matrix_multiply'] = matmul_result
                print(f"Matrix Multiply TOPS: {matmul_result['tops']:.2f}")
            
            # Conv2D
            conv_result = self.measure_conv2d_tops()
            if conv_result:
                self.results['tops_measurements']['conv2d'] = conv_result
                print(f"Conv2D TOPS: {conv_result['tops']:.2f}")
            
            # Linear
            linear_result = self.measure_linear_tops()
            if linear_result:
                self.results['tops_measurements']['linear'] = linear_result
                print(f"Linear TOPS: {linear_result['tops']:.2f}")
            
            # ResNet inference
            resnet_result = self.measure_resnet_inference_tops()
            if resnet_result:
                self.results['tops_measurements']['resnet'] = resnet_result
                print(f"ResNet Inference TOPS: {resnet_result['tops']:.2f}")
        
        # Calculate overall peak TOPS
        tops_values = [r['tops'] for r in self.results['tops_measurements'].values()]
        if tops_values:
            self.results['peak_tops'] = max(tops_values)
            self.results['average_tops'] = np.mean(tops_values)
            
            print("=" * 50)
            print("📊 PERFORMANCE SUMMARY")
            print("=" * 50)
            print(f"🏆 PEAK PERFORMANCE: {self.results['peak_tops']:.2f} TOPS")
            print(f"📊 AVERAGE PERFORMANCE: {self.results['average_tops']:.2f} TOPS")
            print(f"💻 Device: {self.device}")
            
            # Show best performing operation
            best_op = max(self.results['tops_measurements'].items(), key=lambda x: x[1]['tops'])
            print(f"🥇 Best Operation: {best_op[1]['operation']} ({best_op[1]['tops']:.2f} TOPS)")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description='Measure AI performance in TOPS')
    parser.add_argument('--output', type=str, default='tops_results.json',
                        help='Output file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with fewer iterations')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = TOPSBenchmark()
    
    # Adjust runs for quick mode
    if args.quick:
        print("Quick mode: Running with reduced iterations")
        # Override run counts for faster execution
        benchmark.measure_conv2d_tops = lambda: benchmark.measure_conv2d_tops(batch_size=16, runs=10)
        benchmark.measure_linear_tops = lambda: benchmark.measure_linear_tops(batch_size=32, runs=20)
        benchmark.measure_matrix_multiply_tops = lambda: benchmark.measure_matrix_multiply_tops(size=1024, runs=10)
        benchmark.measure_resnet_inference_tops = lambda: benchmark.measure_resnet_inference_tops(batch_size=8, runs=10)
        benchmark.measure_numpy_tops = lambda: benchmark.measure_numpy_tops(size=1000, runs=5)
    
    # Run benchmarks
    results = benchmark.run_all_tops_benchmarks()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()