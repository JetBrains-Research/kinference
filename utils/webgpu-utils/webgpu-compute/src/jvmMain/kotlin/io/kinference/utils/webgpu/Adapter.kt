package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.*
import io.kinference.utils.wgpu.jnr.*

actual class Adapter(private val wgpuAdapter: WGPUAdapter) {
    actual val limits: SupportedLimits
        get() {
            val limits = WGPUSupportedLimits.allocateDirect()
            WebGPUInstance.wgpuNative.wgpuAdapterGetLimits(wgpuAdapter, limits.getPointerTo())
            return SupportedLimits(limits)
        }

    actual suspend fun requestDevice(descriptor: DeviceDescriptor): Device {
        var wgpuDevice: WGPUDevice? = null
        var wgpuError: Exception? = null
        var wgpuStatus: WGPURequestDeviceStatus? = null

        WebGPUInstance.wgpuNative.wgpuAdapterRequestDevice(
            wgpuAdapter,
            descriptor.getPointerTo(),
            { status, device, message, _ ->
                wgpuStatus = status
                if (status == WGPURequestDeviceStatus.Success) {
                    wgpuDevice = device
                } else if (!message.isNullptr) {
                    wgpuError = RuntimeException(message.getString())
                }
            },
            nullptr,
        )
        return wgpuDevice?.let { Device(it) }
            ?: throw wgpuError ?: error("requestDevice() failed: status $wgpuStatus")
    }
}
