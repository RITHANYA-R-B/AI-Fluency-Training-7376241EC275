"""
Day 4 Assignment:
Local AI Study Assistant on a Student Laptop

Estimates model weight memory, KV-cache memory and total memory.
Also compares context lengths and quantization levels.

Scenario:
- Student laptop
- System RAM: detected automatically
- GPU VRAM: detected automatically where possible
"""

import os
import platform
import subprocess
import re


# ============================================================
# MEMORY ESTIMATION CONSTANTS
# ============================================================

BYTES_PER_PARAM = {
    "FP16": 2.00,
    "Q8_0": 1.00,
    "Q6_K": 0.81,
    "Q5_K_M": 0.68,
    "Q4_K_M": 0.57,
    "Q3_K_M": 0.43,
}

KV_GB_PER_B_PER_1K = 0.02
OVERHEAD = 1.10


# ============================================================
# MEMORY ESTIMATION
# ============================================================

def estimate(params_b, precision="Q4_K_M", context_k=8):
    """
    Estimate:
        weights
        KV cache
        total memory

    params_b   = model parameters in billions
    precision  = quantization / precision
    context_k  = context length in thousands of tokens
    """

    if precision not in BYTES_PER_PARAM:
        raise ValueError(f"Unknown precision: {precision}")

    weights_gb = params_b * BYTES_PER_PARAM[precision]

    kv_gb = (
        params_b
        * context_k
        * KV_GB_PER_B_PER_1K
    )

    total_gb = (
        weights_gb + kv_gb
    ) * OVERHEAD

    return weights_gb, kv_gb, total_gb


def verdict(total_gb, available_gb):
    """
    Give a simple fit verdict.
    """

    if total_gb <= available_gb * 0.70:
        return "fits comfortably"

    if total_gb <= available_gb:
        return "fits, but tight"

    return "does NOT fit"


# ============================================================
# HARDWARE DETECTION
# ============================================================

def get_system_ram_gb():
    """
    Detect total physical system RAM on Windows.
    """

    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
            ],
            capture_output=True,
            text=True,
            check=True
        )

        bytes_value = int(result.stdout.strip())
        return bytes_value / (1024 ** 3)

    except Exception:
        return None


def get_gpu_info():
    """
    Try to detect GPU name and adapter memory using PowerShell.
    """

    try:
        command = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, AdapterRAM | "
            "ConvertTo-Json"
        )

        result = subprocess.run(
            ["powershell", "-Command", command],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()

        if not output:
            return []

        import json
        data = json.loads(output)

        if isinstance(data, dict):
            data = [data]

        gpus = []

        for gpu in data:
            name = gpu.get("Name", "Unknown GPU")
            adapter_ram = gpu.get("AdapterRAM")

            if adapter_ram:
                vram_gb = adapter_ram / (1024 ** 3)
            else:
                vram_gb = None

            gpus.append({
                "name": name,
                "vram_gb": vram_gb
            })

        return gpus

    except Exception:
        return []


# ============================================================
# DISPLAY HARDWARE INFORMATION
# ============================================================

def print_hardware():

    print("=" * 72)
    print("                    SYSTEM HARDWARE DETECTION")
    print("=" * 72)

    print(f"OS:                 {platform.system()} {platform.machine()}")
    print(f"Processor:          {platform.processor()}")

    system_ram = get_system_ram_gb()

    if system_ram:
        print(f"System RAM:         {system_ram:.1f} GB")
    else:
        print("System RAM:         Detection failed")

    gpus = get_gpu_info()

    if gpus:
        for gpu in gpus:
            if gpu["vram_gb"]:
                print(
                    f"GPU:                "
                    f"{gpu['name']} ({gpu['vram_gb']:.1f} GB VRAM)"
                )
            else:
                print(f"GPU:                {gpu['name']}")
    else:
        print("GPU:                Detection failed")

    print("=" * 72)

    return system_ram, gpus


# ============================================================
# MODEL REPORT
# ============================================================

def report(
    name,
    params_b,
    precision,
    context_k,
    system_ram_gb,
    gpu_vram_gb
):

    weights, kv, total = estimate(
        params_b,
        precision,
        context_k
    )

    system_result = verdict(
        total,
        system_ram_gb
    )

    gpu_result = verdict(
        total,
        gpu_vram_gb
    )

    print(
        f"{name:<24}"
        f"{precision:<10}"
        f"{params_b:>5.1f}B  "
        f"ctx {context_k:>3}K  "
        f"weights {weights:>6.2f} GB  "
        f"kv {kv:>6.2f} GB  "
        f"total {total:>6.2f} GB"
    )

    print(
        f"    System RAM {system_ram_gb:.1f} GB -> "
        f"{system_result}"
    )

    print(
        f"    GPU VRAM    {gpu_vram_gb:.1f} GB -> "
        f"{gpu_result}"
    )

    return weights, kv, total


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    system_ram, gpus = print_hardware()

    # --------------------------------------------------------
    # Fallback values if detection fails
    # --------------------------------------------------------

    if system_ram is None:
        system_ram = 7.8

    if gpus and gpus[0]["vram_gb"]:
        gpu_vram = gpus[0]["vram_gb"]

        # Keep the same 0.9 GB budget idea used in your
        # current lab program for GPU safety margin.
        gpu_budget = gpu_vram * 0.90
    else:
        gpu_vram = 0.0
        gpu_budget = 0.0

    print()
    print("=" * 72)
    print("                         MEMORY BUDGET")
    print("=" * 72)

    print(f"System RAM budget:     {system_ram:.1f} GB")

    if gpu_budget > 0:
        print(
            f"GPU VRAM detected:     {gpu_vram:.1f} GB"
        )
        print(
            f"GPU safety budget:     {gpu_budget:.1f} GB"
        )
    else:
        print("GPU VRAM budget:       Not available")

    print("=" * 72)


    # ========================================================
    # 1. FOUR MODEL CONFIGURATIONS
    # ========================================================

    print()
    print("=" * 72)
    print("                 FOUR MODEL CONFIGURATIONS")
    print("=" * 72)

    configurations = [
        ("Small model", 1.5, "Q4_K_M", 8),
        ("Mid model", 8.0, "Q4_K_M", 8),
        ("Mid model Q5", 8.0, "Q5_K_M", 8),
        ("Mid model long ctx", 8.0, "Q4_K_M", 32),
    ]

    for name, params, precision, context in configurations:

        report(
            name,
            params,
            precision,
            context,
            system_ram,
            gpu_budget
        )


    # ========================================================
    # 2. CONTEXT LENGTH EXPERIMENT
    # ========================================================

    print()
    print("=" * 72)
    print("              SAME 8B MODEL - CONTEXT EXPERIMENT")
    print("=" * 72)

    for context in [4, 8, 32, 128]:

        weights, kv, total = estimate(
            8.0,
            "Q4_K_M",
            context
        )

        print(
            f"8B Q4_K_M  "
            f"ctx {context:>3}K  "
            f"weights {weights:>6.2f} GB  "
            f"kv {kv:>6.2f} GB  "
            f"total {total:>6.2f} GB  "
            f"RAM: {verdict(total, system_ram)}"
        )


    # ========================================================
    # 3. QUANTIZATION EXPERIMENT
    # ========================================================

    print()
    print("=" * 72)
    print("              SAME 8B MODEL - QUANTIZATION")
    print("=" * 72)

    for precision in [
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q8_0",
        "FP16"
    ]:

        weights, kv, total = estimate(
            8.0,
            precision,
            8
        )

        print(
            f"8B {precision:<8} "
            f"ctx 8K  "
            f"weights {weights:>6.2f} GB  "
            f"kv {kv:>6.2f} GB  "
            f"total {total:>6.2f} GB  "
            f"RAM: {verdict(total, system_ram)}"
        )


    # ========================================================
    # 4. FINAL SCENARIO SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("                    SCENARIO SUMMARY")
    print("=" * 72)

    print(
        "Scenario: Local AI Study Assistant on a student laptop"
    )

    print(
        f"System RAM available: {system_ram:.1f} GB"
    )

    if gpu_budget > 0:
        print(
            f"GPU VRAM safety budget: {gpu_budget:.1f} GB"
        )

    print()
    print(
        "The laptop cannot comfortably run the tested models "
        "within the dedicated GPU VRAM budget."
    )

    print(
        "CPU/system-RAM execution must therefore be considered "
        "separately from GPU-only execution."
    )

    print("=" * 72)