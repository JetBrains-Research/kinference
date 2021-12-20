package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.MemoryMode
import io.kinference.utils.wgpu.internal.createPointerTo
import io.kinference.utils.wgpu.jnr.*
import io.kinference.utils.wgpu.jnr.utils.loadWgsl

actual class BindGroupDescriptor actual constructor(
    layout: BindGroupLayout,
    entries: List<BindGroupEntry>
) : WGPUBindGroupDescriptor(MemoryMode.Direct) {
    init {
        this.layout = layout.wgpuBindGroupLayout
        this.entries.set(entries.toTypedArray())
        this.entryCount = entries.size.toLong()
    }
}

actual class BindGroupLayoutDescriptor actual constructor(
    entries: List<BindGroupLayoutEntry>
) : WGPUBindGroupLayoutDescriptor(MemoryMode.Direct) {
    init {
        this.entries.set(entries.toTypedArray())
        this.entryCount = entries.size.toLong()
    }
}

actual class BufferDescriptor actual constructor(
    size: Int,
    usage: BufferUsageFlags,
    mappedAtCreation: kotlin.Boolean
) : WGPUBufferDescriptor(MemoryMode.Direct) {
    init {
        this.size = size.toLong()
        this.usage = usage.value
        this.mappedAtCreation = mappedAtCreation
    }
}

actual class CommandBufferDescriptor : WGPUCommandBufferDescriptor(MemoryMode.Direct)

actual class CommandEncoderDescriptor : WGPUCommandEncoderDescriptor(MemoryMode.Direct)

actual class ComputePassDescriptor : WGPUComputePassDescriptor(MemoryMode.Direct)

actual class ComputePipelineDescriptor actual constructor(
    layout: PipelineLayout,
    compute: ProgrammableStage
) : WGPUComputePipelineDescriptor(MemoryMode.Direct) {
    init {
        this.layout = layout.wgpuPipelineLayout
        this.compute.apply {
            module = compute.module.wgpuShaderModule
            entryPoint = compute.entryPoint.createPointerTo()
        }
    }
}

actual class DeviceDescriptor actual constructor(requiredLimits: Limits) : WGPUDeviceDescriptor(MemoryMode.Direct) {
    init {
        this.requiredLimits.set(WGPURequiredLimits.allocateDirect().apply {
            limits.apply {
                maxBindGroups = requiredLimits.maxBindGroups?.toLong() ?: 0
                maxDynamicStorageBuffersPerPipelineLayout =
                    requiredLimits.maxDynamicStorageBuffersPerPipelineLayout?.toLong() ?: 0
                maxStorageBuffersPerShaderStage = requiredLimits.maxStorageBuffersPerShaderStage?.toLong() ?: 0
                maxStorageBufferBindingSize = requiredLimits.maxStorageBufferBindingSize?.toLong() ?: 0
                minStorageBufferOffsetAlignment = requiredLimits.minStorageBufferOffsetAlignment?.toLong() ?: 0
                maxComputeWorkgroupStorageSize = requiredLimits.maxComputeWorkgroupStorageSize?.toLong() ?: 0
                maxComputeInvocationsPerWorkgroup = requiredLimits.maxComputeInvocationsPerWorkgroup?.toLong() ?: 0
                maxComputeWorkgroupSizeX = requiredLimits.maxComputeWorkgroupSizeX?.toLong() ?: 0
                maxComputeWorkgroupSizeY = requiredLimits.maxComputeWorkgroupSizeY?.toLong() ?: 0
                maxComputeWorkgroupSizeZ = requiredLimits.maxComputeWorkgroupSizeZ?.toLong() ?: 0
                maxComputeWorkgroupsPerDimension = requiredLimits.maxComputeWorkgroupsPerDimension?.toLong() ?: 0
            }
        })
    }
}

actual class PipelineLayoutDescriptor actual constructor(
    bindGroupLayouts: List<BindGroupLayout>
) : WGPUPipelineLayoutDescriptor(MemoryMode.Direct) {
    init {
        this.bindGroupLayouts = bindGroupLayouts.map { it.wgpuBindGroupLayout.address() }.toLongArray().createPointerTo()
        this.bindGroupLayoutCount = bindGroupLayouts.size.toLong()
    }
}

actual class ShaderModuleDescriptor actual constructor(code: String) {
    val descriptor = loadWgsl(code)
}
