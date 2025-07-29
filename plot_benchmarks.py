import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# File paths
amd64_file = 'docs/results.txt'
#aarch64_file = 'build/results/jmh/aarch64.txt'

# Function to parse benchmark file
def parse_benchmark_file(file_path, system_name):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    data_lines = lines[1:]

    # Parse data
    data = []
    for line in data_lines:
        if not line.strip():
            continue

        parts = line.split()
        benchmark_full = parts[0]
        size = int(parts[1])
        score = float(parts[4].replace(',', '.'))  # Handle comma as decimal separator

        # Split benchmark name into benchmark type and implementation
        benchmark_parts = benchmark_full.split('.')
        benchmark_type = benchmark_parts[0]
        implementation = benchmark_parts[1]

        data.append({
            'benchmark_type': benchmark_type,
            'implementation': implementation,
            'size': size,
            'score': score,
            'system': system_name
        })

    return data

# Parse both files
amd64_data = parse_benchmark_file(amd64_file, 'amd64')
#aarch64_data = parse_benchmark_file(aarch64_file, 'aarch64')

# Combine data
combined_data = amd64_data

# Organize data by benchmark type
benchmark_data = {}
for entry in combined_data:
    benchmark_type = entry['benchmark_type']
    if benchmark_type not in benchmark_data:
        benchmark_data[benchmark_type] = []
    benchmark_data[benchmark_type].append(entry)

# Create output directory for PNG files
os.makedirs('docs/benchmark_plots', exist_ok=True)

print("\nGenerating benchmark plots...")

# Process each benchmark type
for benchmark_type, data in benchmark_data.items():
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Get unique array sizes
    sizes = sorted(df['size'].unique())

    # Create a new figure for each benchmark
    # Adjust figure width based on number of array sizes
    num_sizes = len(sizes)
    fig_width = max(12, num_sizes * 3)  # Ensure minimum width of 15, but scale with number of plots
    plt.figure(figsize=(fig_width, 4), dpi=300)

    # Create subplots for each array size in a single row
    for i, size in enumerate(sizes):
        # Filter data for this array size
        size_df = df[df['size'] == size]

        # Create subplot in a single row
        plt.subplot(1, num_sizes, i + 1)

        # Create barplot with system as hue
        sns.barplot(
            x='implementation',
            y='score',
            hue='system',
            data=size_df,
        )

        # Set up the subplot
        plt.title(f"Array Size: {size}")
        # plt.yscale('log')  # Logarithmic scale for y-axis
        plt.ylabel('Score (ops/s)')
        plt.xticks(rotation=45)
        plt.legend(title='System')
        plt.grid(True, which="both", ls="-", alpha=0.2)

    # Set up the overall plot
    plt.suptitle(benchmark_type, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save the plot as PNG
    output_file = f"docs/benchmark_plots/{benchmark_type}.png"
    plt.savefig(output_file)
    plt.close()

    print(f"Created: {output_file}")

    # Also output data in text format
    print(f"\n=== {benchmark_type} ===")
    print("Implementation, System, Size, Score (ops/s)")

    # Sort data by implementation, system, and array size
    sorted_data = sorted(data, key=lambda x: (
        x['implementation'], 
        x['system'], 
        x['size']
    ))

    for entry in sorted_data:
        print(f"{entry['implementation']}, {entry['system']}, {entry['size']}, {entry['score']}")

print("\nPNG charts have been created in the 'docs/benchmark_plots' directory.")
