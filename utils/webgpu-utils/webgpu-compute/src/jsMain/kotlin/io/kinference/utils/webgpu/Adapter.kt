package io.kinference.utils.webgpu

import kotlinx.coroutines.await
import kotlin.js.Promise

actual class Adapter(val gpuAdapter: GPUAdapter) {
    actual val limits
        get() = gpuAdapter.limits

    actual suspend fun requestDevice(descriptor: DeviceDescriptor): Device =
        Device(gpuAdapter.requestDevice(descriptor).await())
}

external class GPUAdapter {
    val limits: GPUSupportedLimits

    fun requestDevice(descriptor: DeviceDescriptor = definedExternally): Promise<GPUDevice>
}
