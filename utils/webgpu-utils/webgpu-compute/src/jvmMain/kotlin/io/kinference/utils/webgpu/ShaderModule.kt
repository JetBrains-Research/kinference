package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.jnr.WGPUShaderModule

actual class ShaderModule(val wgpuShaderModule: WGPUShaderModule) {
    actual suspend fun compilationInfo(): CompilationInfo {
        TODO("Not yet implemented")
    }
}
