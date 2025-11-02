# AI Model Memory-Traffic Simulator
# Week 3: Data Movement Energy in AI Inference

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 256                          # matrix size
energy_dram = 100e-12            # 100 pJ per DRAM access
energy_sram = 10e-12             # 10 pJ per SRAM access
energy_compute = 5e-12           # 5 pJ per multiply-accumulate (MAC)

def simulate_matrix_mult(N, cache_reuse=True):
    """Estimate compute and memory energy for NxN matrix multiply."""
    # compute operations
    macs = N ** 3
    
    # memory access estimates
    if cache_reuse:
        # assume 90% of reads come from fast SRAM cache
        reads_sram = 0.9 * (2 * N**2)
        reads_dram = 0.1 * (2 * N**2)
    else:
        # no reuse, all reads from slow DRAM
        reads_sram = 0
        reads_dram = 2 * N**2
    
    # compute energy usage
    compute_energy = macs * energy_compute
    memory_energy = (reads_sram * energy_sram) + (reads_dram * energy_dram)
    
    return compute_energy, memory_energy

def run_simulation():
    """Run the memory traffic simulation and generate results."""
    # Run simulation for two cases
    compute_reuse, memory_reuse = simulate_matrix_mult(N, True)
    compute_no_reuse, memory_no_reuse = simulate_matrix_mult(N, False)
    
    # Prepare chart
    labels = ['Compute', 'Memory (with reuse)', 'Memory (no reuse)']
    values = [compute_reuse, memory_reuse, memory_no_reuse]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, np.array(values)*1e6)  # convert J → µJ
    plt.ylabel('Energy (µJ)')
    plt.title('Compute vs Memory Energy for AI Inference')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Results
    print(f"Matrix size: {N}x{N}")
    print(f"Energy with cache reuse: {memory_reuse/compute_reuse:.1f}x compute energy")
    print(f"Energy without cache reuse: {memory_no_reuse/compute_no_reuse:.1f}x compute energy")
    print(f"\nDetailed Results:")
    print(f"  Compute energy: {compute_reuse*1e6:.2f} µJ")
    print(f"  Memory energy (with reuse): {memory_reuse*1e6:.2f} µJ")
    print(f"  Memory energy (no reuse): {memory_no_reuse*1e6:.2f} µJ")

def run_size_comparison():
    """Compare energy consumption across different matrix sizes."""
    sizes = [64, 128, 256, 512, 1024]
    compute_energies = []
    memory_energies_reuse = []
    memory_energies_no_reuse = []
    
    for size in sizes:
        comp_reuse, mem_reuse = simulate_matrix_mult(size, True)
        comp_no_reuse, mem_no_reuse = simulate_matrix_mult(size, False)
        
        compute_energies.append(comp_reuse * 1e6)  # Convert to µJ
        memory_energies_reuse.append(mem_reuse * 1e6)
        memory_energies_no_reuse.append(mem_no_reuse * 1e6)
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(sizes))
    width = 0.25
    
    plt.bar(x - width, compute_energies, width, label='Compute Energy', alpha=0.8)
    plt.bar(x, memory_energies_reuse, width, label='Memory Energy (with reuse)', alpha=0.8)
    plt.bar(x + width, memory_energies_no_reuse, width, label='Memory Energy (no reuse)', alpha=0.8)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Energy (µJ)')
    plt.title('Energy Consumption vs Matrix Size')
    plt.xticks(x, [f'{s}x{s}' for s in sizes])
    plt.legend()
    plt.yscale('log')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 AI Model Memory-Traffic Simulator")
    print("=" * 40)
    
    # Run basic simulation
    run_simulation()
    
    print("\n" + "=" * 40)
    print("📊 Size Comparison Analysis")
    
    # Run size comparison
    run_size_comparison()