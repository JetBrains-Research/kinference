package io.kinference.utils.webgpu

expect class ShaderModule {
    suspend fun compilationInfo(): CompilationInfo
}
