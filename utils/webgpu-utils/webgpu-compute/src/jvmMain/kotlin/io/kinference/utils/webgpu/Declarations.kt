package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.jnr.*

actual typealias BufferDynamicOffset = Long

actual class BindGroup(val wgpuBindGroup: WGPUBindGroup)
actual class BindGroupLayout(val wgpuBindGroupLayout: WGPUBindGroupLayout)
actual class CommandBuffer(val wgpuCommandBuffer: WGPUCommandBuffer)
actual class ComputePipeline(val wgpuComputePipeline: WGPUComputePipeline)
actual class PipelineLayout(val wgpuPipelineLayout: WGPUPipelineLayout)
