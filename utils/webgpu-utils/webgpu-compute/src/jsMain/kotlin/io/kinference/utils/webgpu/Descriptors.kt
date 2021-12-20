package io.kinference.utils.webgpu

import kotlin.js.Json

actual class BindGroupDescriptor actual constructor(
    val layout: BindGroupLayout,
    entries: List<BindGroupEntry>,
) {
    val entries = entries.toTypedArray()
}

actual class BindGroupLayoutDescriptor actual constructor(
    entries: List<BindGroupLayoutEntry>,
) {
    val entries = entries.toTypedArray()
}

actual class BufferDescriptor actual constructor(
    val size: Int,
    usage: BufferUsageFlags,
    val mappedAtCreation: Boolean,
) {
    val usage = usage.value
}

actual class CommandBufferDescriptor

actual class CommandEncoderDescriptor

actual class ComputePassDescriptor

actual class ComputePipelineDescriptor actual constructor(
    val layout: PipelineLayout,
    val compute: ProgrammableStage,
)

actual class DeviceDescriptor actual constructor(requiredLimits: Limits) {
    val requiredLimits: Json = requiredLimits.record
}

actual class PipelineLayoutDescriptor actual constructor(
    bindGroupLayouts: List<BindGroupLayout>,
) {
    val bindGroupLayouts = bindGroupLayouts.toTypedArray()
}

actual class ShaderModuleDescriptor actual constructor(
    val code: String,
)
