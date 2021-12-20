package io.kinference.utils.webgpu

import kotlinx.coroutines.await
import kotlin.js.Promise

actual object WebGPUInstance {
    val gpu: GPU = let {
        val gpuInstance = js("navigator.gpu")
        if (gpuInstance == undefined) {
            error("WebGPU not enabled")
        }
        gpuInstance as GPU
    }

    actual suspend fun requestAdapter(options: RequestAdapterOptions): Adapter {
        gpu.requestAdapter(options).await()?.let { return Adapter(it) } ?: error("requestAdapter() failed")
    }
}

external class GPU {
    fun requestAdapter(options: RequestAdapterOptions): Promise<GPUAdapter?>
}
