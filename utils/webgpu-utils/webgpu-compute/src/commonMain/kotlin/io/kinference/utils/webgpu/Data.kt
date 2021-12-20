package io.kinference.utils.webgpu

import kotlin.math.max

expect class BindGroupEntry(
    binding: Int,
    resource: BufferBinding,
)

expect class BindGroupLayoutEntry(
    binding: Int,
    buffer: BufferBindingLayout,
)

expect class BufferBinding(
    buffer: Buffer,
    offset: Int = 0,
    size: Int = max(0, buffer.size - offset),
)

expect class BufferBindingLayout(
    type: BufferBindingType,
    hasDynamicOffset: Boolean = false,
    minBindingSize: Int = 0
)

expect class BufferUsageFlags(
    vararg flags: BufferUsage
)

expect class CompilationInfo {
    val messages: List<CompilationMessage>
}

expect class CompilationMessage {
    val message: String
    val type: CompilationMessageType
    val lineNum: Int
    val linePos: Int
    val offset: Int
    val length: Int
}

expect class Limits(
    maxBindGroups: Int? = null,
    maxDynamicStorageBuffersPerPipelineLayout: Int? = null,
    maxStorageBuffersPerShaderStage: Int? = null,
    maxStorageBufferBindingSize: Int? = null,
    minStorageBufferOffsetAlignment: Int? = null,
    maxComputeWorkgroupStorageSize: Int? = null,
    maxComputeInvocationsPerWorkgroup: Int? = null,
    maxComputeWorkgroupSizeX: Int? = null,
    maxComputeWorkgroupSizeY: Int? = null,
    maxComputeWorkgroupSizeZ: Int? = null,
    maxComputeWorkgroupsPerDimension: Int? = null,
)

expect class MapModeFlags(
    vararg flags: MapMode,
)

expect class ProgrammableStage(
    module: ShaderModule,
    entryPoint: String,
)

expect class RequestAdapterOptions(
    powerPreference: PowerPreference = PowerPreference.HighPerformance,
    forceFallbackAdapter: Boolean = false,
)

expect class SupportedLimits {
    val maxBindGroups: Int
    val maxDynamicStorageBuffersPerPipelineLayout: Int
    val maxStorageBuffersPerShaderStage: Int
    val maxStorageBufferBindingSize: Int
    val minStorageBufferOffsetAlignment: Int
    val maxComputeWorkgroupStorageSize: Int
    val maxComputeInvocationsPerWorkgroup: Int
    val maxComputeWorkgroupSizeX: Int
    val maxComputeWorkgroupSizeY: Int
    val maxComputeWorkgroupSizeZ: Int
    val maxComputeWorkgroupsPerDimension: Int
}

fun SupportedLimits.asString() =
    """
        |{
        |   maxBindGroups: $maxBindGroups,
        |   maxDynamicStorageBuffersPerPipelineLayout: $maxDynamicStorageBuffersPerPipelineLayout,
        |   maxStorageBuffersPerShaderStage: $maxStorageBuffersPerShaderStage,
        |   maxStorageBufferBindingSize: $maxStorageBufferBindingSize,
        |   minStorageBufferOffsetAlignment: $minStorageBufferOffsetAlignment,
        |   maxComputeWorkgroupStorageSize: $maxComputeWorkgroupStorageSize,
        |   maxComputeInvocationsPerWorkgroup: $maxComputeInvocationsPerWorkgroup,
        |   maxComputeWorkgroupSizeX: $maxComputeWorkgroupSizeX,
        |   maxComputeWorkgroupSizeY: $maxComputeWorkgroupSizeY,
        |   maxComputeWorkgroupSizeZ: $maxComputeWorkgroupSizeZ,
        |   maxComputeWorkgroupsPerDimension: $maxComputeWorkgroupsPerDimension,
        |}
    """.trimMargin()
