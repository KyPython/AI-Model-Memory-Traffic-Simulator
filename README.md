# 🧠 AI Model Memory-Traffic Simulator

**Understanding Memory Hierarchy and Data Movement Energy in AI Computing**

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project simulates and analyzes the energy consumption patterns in AI inference, focusing on the critical relationship between compute energy and memory access energy. It demonstrates why "moving bits" often consumes more energy than "computing with them" in modern AI systems.

### 🎯 Key Learning Objectives

- **Memory Hierarchy Understanding**: Explore the energy costs of different memory levels (SRAM vs DRAM)
- **Cache Optimization**: See how data reuse dramatically reduces energy consumption  
- **The Memory Wall**: Understand why data movement dominates AI chip power consumption
- **Architecture Impact**: Compare energy patterns across different matrix sizes

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib
```

### Running the Simulation

```bash
python memory_traffic_simulator.py
```

## 📊 What You'll Discover

### Energy Breakdown Analysis
The simulator reveals that:
- **With caching**: Memory energy ≈ 1-2x compute energy
- **Without caching**: Memory energy > 10x compute energy
- **Scale impact**: Larger matrices amplify the memory wall problem

### Visual Results
The tool generates two key visualizations:
1. **Energy Comparison Chart**: Shows compute vs memory energy for a single matrix size
2. **Scaling Analysis**: Demonstrates how energy consumption grows with matrix size

## 🔬 Technical Details

### Energy Model Parameters
- **DRAM Access**: 100 pJ per access (slow, high energy)
- **SRAM Access**: 10 pJ per access (fast, low energy)  
- **MAC Operation**: 5 pJ per multiply-accumulate

### Cache Simulation
- **With Reuse**: 90% of reads from SRAM, 10% from DRAM
- **No Reuse**: 100% of reads from DRAM (worst case)

### Matrix Operations
Simulates `N×N` matrix multiplication with `N³` operations and `2×N²` memory reads.

## 📈 Example Output

```
🚀 AI Model Memory-Traffic Simulator
========================================
Matrix size: 256x256
Energy with cache reuse: 1.3x compute energy
Energy without cache reuse: 12.8x compute energy

Detailed Results:
  Compute energy: 83.89 µJ
  Memory energy (with reuse): 109.23 µJ  
  Memory energy (no reuse): 1073.74 µJ
```

## 🎓 Educational Context

This simulator is part of **Week 3: Memory Systems and Data Movement in AI Computing** - a study of how memory architecture limits AI performance and how engineers optimize bandwidth, latency, and energy.

### Key Concepts Demonstrated
- **Memory Hierarchy**: registers → cache → DRAM → storage
- **Bandwidth vs Latency**: Trading speed for capacity
- **Power Scaling**: P ∝ C × V² × f for data movement
- **Architectural Solutions**: Why we need systolic arrays and tensor cores

## 🛠️ Extending the Simulator

### Customization Options
- Modify energy parameters for different hardware
- Adjust cache hit ratios for various algorithms
- Add more memory levels (L1, L2, L3 cache)
- Implement different matrix algorithms

### Advanced Features Ideas
- Memory bandwidth limitations
- Parallel processing simulation  
- Quantization impact on memory usage
- Different neural network layer types

## 📚 Related Concepts

- **AI Accelerators**: TPUs, GPUs, and specialized chips
- **Memory Technologies**: HBM, GDDR, processing-in-memory
- **Optimization Techniques**: Tiling, data layout, compression
- **Architecture Design**: Von Neumann bottleneck solutions

## 🤝 Contributing

Feel free to:
- Add new memory models
- Implement different AI workloads
- Improve visualization
- Add more realistic energy models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Learn More

- [Memory Wall Problem in Computing](https://en.wikipedia.org/wiki/Memory_wall)
- [AI Chip Architecture Basics](https://semiengineering.com/ai-chip-architectures/)
- [Energy-Efficient AI Computing](https://arxiv.org/abs/1907.10895)

---

*Built to demonstrate the fundamental energy tradeoffs in AI computing systems* 🚀