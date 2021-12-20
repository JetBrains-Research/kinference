package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.MemoryMode
import io.kinference.utils.wgpu.internal.getString
import io.kinference.utils.wgpu.jnr.*

actual class BindGroupEntry actual constructor(
    binding: Int,
    resource: BufferBinding
) : WGPUBindGroupEntry(MemoryMode.Direct) {
    init {
        this.binding = binding.toLong()
        this.buffer = resource.buffer.wgpuBuffer
        this.offset = resource.offset.toLong()
        this.size = resource.size.toLong()
    }
}

actual class BindGroupLayoutEntry actual constructor(
    binding: Int,
    buffer: BufferBindingLayout
) : WGPUBindGroupLayoutEntry(MemoryMode.Direct) {
    init {
        this.binding = binding.toLong()
        this.visibility = WGPUShaderStage.Compute.value.toLong()
        this.buffer.apply {
            type = buffer.type
            hasDynamicOffset = buffer.hasDynamicOffset
            minBindingSize = buffer.minBindingSize.toLong()
        }
    }
}

actual class BufferBinding actual constructor(
    val buffer: Buffer,
    val offset: Int,
    val size: Int
)

actual class BufferBindingLayout actual constructor(
    val type: BufferBindingType,
    val hasDynamicOffset: Boolean,
    val minBindingSize: Int
)

actual class BufferUsageFlags(val value: Long) {
    actual constructor(vararg flags: BufferUsage) : this(flags.map { it.value }.fold(0, Int::or).toLong())
}

actual class CompilationInfo(wgpuCompilationInfo: WGPUCompilationInfo) {
    actual val messages: List<CompilationMessage> =
        wgpuCompilationInfo.messages.get(wgpuCompilationInfo.messageCount.toInt()).map { CompilationMessage(it) }
}

actual class CompilationMessage(private val wgpuCompilationMessage: WGPUCompilationMessage) {
    actual val message: String
        get() = wgpuCompilationMessage.message.getString()
    actual val type: CompilationMessageType
        get() = wgpuCompilationMessage.type
    actual val lineNum: Int
        get() = wgpuCompilationMessage.lineNum.toInt()
    actual val linePos: Int
        get() = wgpuCompilationMessage.linePos.toInt()
    actual val offset: Int
        get() = wgpuCompilationMessage.offset.toInt()
    actual val length: Int
        get() = wgpuCompilationMessage.length.toInt()
}

actual class Limits actual constructor(
    val maxBindGroups: Int?,
    val maxDynamicStorageBuffersPerPipelineLayout: Int?,
    val maxStorageBuffersPerShaderStage: Int?,
    val maxStorageBufferBindingSize: Int?,
    val minStorageBufferOffsetAlignment: Int?,
    val maxComputeWorkgroupStorageSize: Int?,
    val maxComputeInvocationsPerWorkgroup: Int?,
    val maxComputeWorkgroupSizeX: Int?,
    val maxComputeWorkgroupSizeY: Int?,
    val maxComputeWorkgroupSizeZ: Int?,
    val maxComputeWorkgroupsPerDimension: Int?
)

actual class MapModeFlags(val value: Long) {
    actual constructor(vararg flags: MapMode) : this(flags.map { it.value }.fold(0, Int::or).toLong())
}

actual class ProgrammableStage actual constructor(
    val module: ShaderModule,
    val entryPoint: String
)

actual class RequestAdapterOptions actual constructor(
    powerPreference: PowerPreference,
    forceFallbackAdapter: kotlin.Boolean
) : WGPURequestAdapterOptions(MemoryMode.Direct) {
    init {
        this.powerPreference = powerPreference
        this.forceFallbackAdapter = forceFallbackAdapter
    }
}

actual class SupportedLimits(private val wgpuSupportedLimits: WGPUSupportedLimits) {
    actual val maxBindGroups: Int
        get() = wgpuSupportedLimits.limits.maxBindGroups.toInt()
    actual val maxDynamicStorageBuffersPerPipelineLayout: Int
        get() = wgpuSupportedLimits.limits.maxDynamicStorageBuffersPerPipelineLayout.toInt()
    actual val maxStorageBuffersPerShaderStage: Int
        get() = wgpuSupportedLimits.limits.maxStorageBuffersPerShaderStage.toInt()
    actual val maxStorageBufferBindingSize: Int
        get() = wgpuSupportedLimits.limits.maxStorageBufferBindingSize.toInt()
    actual val minStorageBufferOffsetAlignment: Int
        get() = wgpuSupportedLimits.limits.minStorageBufferOffsetAlignment.toInt()
    actual val maxComputeWorkgroupStorageSize: Int
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupStorageSize.toInt()
    actual val maxComputeInvocationsPerWorkgroup: Int
        get() = wgpuSupportedLimits.limits.maxComputeInvocationsPerWorkgroup.toInt()
    actual val maxComputeWorkgroupSizeX: Int
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeX.toInt()
    actual val maxComputeWorkgroupSizeY: Int
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeY.toInt()
    actual val maxComputeWorkgroupSizeZ: Int
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeZ.toInt()
    actual val maxComputeWorkgroupsPerDimension: Int
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupsPerDimension.toInt()
}
