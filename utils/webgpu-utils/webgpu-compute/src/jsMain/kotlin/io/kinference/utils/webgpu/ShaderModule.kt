package io.kinference.utils.webgpu

import kotlinx.coroutines.await
import kotlin.js.Promise

actual class ShaderModule(val gpuShaderModule: GPUShaderModule) {
    actual suspend fun compilationInfo(): CompilationInfo =
        CompilationInfo(gpuShaderModule.compilationInfo().await())
}

external class GPUShaderModule {
    fun compilationInfo(): Promise<GPUCompilationInfo>
}
