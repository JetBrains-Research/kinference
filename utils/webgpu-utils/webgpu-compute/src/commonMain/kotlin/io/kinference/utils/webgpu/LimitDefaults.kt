package io.kinference.utils.webgpu

object LimitDefaults {
    const val maxBindGroups: Long = 4
    const val maxDynamicStorageBuffersPerPipelineLayout: Long = 4
    const val maxStorageBuffersPerShaderStage: Long = 8
    const val maxStorageBufferBindingSize: Long = 134217728
    const val minStorageBufferOffsetAlignment: Long = 256
    const val maxComputeWorkgroupStorageSize: Long = 16352
    const val maxComputeInvocationsPerWorkgroup: Long = 256
    const val maxComputeWorkgroupSizeX: Long = 256
    const val maxComputeWorkgroupSizeY: Long = 256
    const val maxComputeWorkgroupSizeZ: Long = 64
    const val maxComputeWorkgroupsPerDimension: Long = 65535
}
