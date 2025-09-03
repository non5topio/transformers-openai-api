#!/usr/bin/env python3
"""
Performance monitoring and optimization script for transformers-openai-api
"""

import psutil
import time
import requests
import json
import argparse
from typing import Dict, List


class PerformanceMonitor:
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.metrics = []
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': time.time()
        }
    
    def test_api_response_time(self) -> Dict:
        """Test API response time"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_url}/v1/engines", timeout=10)
            end_time = time.time()
            return {
                'response_time': end_time - start_time,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'response_time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
    
    def run_monitoring(self, duration: int = 60, interval: int = 5):
        """Run continuous monitoring"""
        print(f"Starting monitoring for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            system_metrics = self.get_system_metrics()
            api_metrics = self.test_api_response_time()
            
            combined_metrics = {**system_metrics, **api_metrics}
            self.metrics.append(combined_metrics)
            
            print(f"CPU: {system_metrics['cpu_percent']:.1f}% | "
                  f"Memory: {system_metrics['memory_percent']:.1f}% | "
                  f"API Response: {api_metrics.get('response_time', 'N/A'):.3f}s")
            
            time.sleep(interval)
    
    def save_metrics(self, filename: str = "performance_metrics.json"):
        """Save metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filename}")
    
    def analyze_metrics(self):
        """Analyze collected metrics"""
        if not self.metrics:
            print("No metrics to analyze")
            return
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_percent'] for m in self.metrics]
        response_times = [m.get('response_time', 0) for m in self.metrics if m.get('success')]
        
        print("\n=== Performance Analysis ===")
        print(f"Average CPU usage: {sum(cpu_values)/len(cpu_values):.1f}%")
        print(f"Peak CPU usage: {max(cpu_values):.1f}%")
        print(f"Average Memory usage: {sum(memory_values)/len(memory_values):.1f}%")
        print(f"Peak Memory usage: {max(memory_values):.1f}%")
        
        if response_times:
            print(f"Average API response time: {sum(response_times)/len(response_times):.3f}s")
            print(f"Slowest API response: {max(response_times):.3f}s")
            print(f"Fastest API response: {min(response_times):.3f}s")
        
        # Performance recommendations
        self.provide_recommendations(cpu_values, memory_values, response_times)
    
    def provide_recommendations(self, cpu_values: List[float], 
                              memory_values: List[float], 
                              response_times: List[float]):
        """Provide performance optimization recommendations"""
        print("\n=== Optimization Recommendations ===")
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        
        if avg_cpu > 80:
            print("⚠️  High CPU usage detected. Consider:")
            print("   - Reducing model size or using quantization")
            print("   - Implementing request queuing")
            print("   - Adding more CPU cores")
        
        if avg_memory > 80:
            print("⚠️  High memory usage detected. Consider:")
            print("   - Using smaller models")
            print("   - Implementing model caching with TTL")
            print("   - Adding more RAM")
        
        if avg_response > 2.0:
            print("⚠️  Slow API responses detected. Consider:")
            print("   - Using GPU acceleration")
            print("   - Implementing model quantization")
            print("   - Adding response caching")
        
        if avg_cpu < 50 and avg_memory < 50:
            print("✅ System resources are well utilized")
        
        if avg_response < 1.0:
            print("✅ API response times are good")


def main():
    parser = argparse.ArgumentParser(description="Monitor transformers-openai-api performance")
    parser.add_argument("--url", default="http://localhost:8001", help="API URL to monitor")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--output", default="performance_metrics.json", help="Output file for metrics")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.url)
    
    try:
        monitor.run_monitoring(args.duration, args.interval)
        monitor.analyze_metrics()
        monitor.save_metrics(args.output)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        monitor.analyze_metrics()
        monitor.save_metrics(args.output)


if __name__ == "__main__":
    main()
