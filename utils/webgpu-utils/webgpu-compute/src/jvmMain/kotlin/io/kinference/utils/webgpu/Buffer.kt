package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.nullptr
import io.kinference.utils.wgpu.jnr.WGPUBuffer
import io.kinference.utils.wgpu.jnr.WGPUBufferMapAsyncStatus
import io.kinference.utils.wgpu.jnr.WGPUDevice

actual class Buffer(val wgpuBuffer: WGPUBuffer, actual val size: Int, val wgpuDevice: WGPUDevice) {
    actual fun destroy() = WebGPUInstance.wgpuNative.wgpuBufferDestroy(wgpuBuffer)

    actual fun getMappedRange(offset: Int, size: Int): BufferData {
        val actualSize = if (size >= 0) size else maxOf(0, this.size - offset)
        return BufferData(
            WebGPUInstance.wgpuNative.wgpuBufferGetMappedRange(wgpuBuffer, offset.toLong(), actualSize.toLong()),
            actualSize
        )
    }

    actual suspend fun mapAsync(mode: MapModeFlags, offset: Int, size: Int) {
        val actualSize = if (size >= 0) size else maxOf(0, this.size - offset)
        var wgpuStatus: WGPUBufferMapAsyncStatus = WGPUBufferMapAsyncStatus.Unknown
        WebGPUInstance.wgpuNative.wgpuBufferMapAsync(
            wgpuBuffer, mode.value, offset.toLong(), actualSize.toLong(),
            { status, _ ->
                wgpuStatus = status
            },
            nullptr
        )
        WebGPUInstance.wgpuNative.wgpuDevicePoll(wgpuDevice, force_wait = true)
        if (wgpuStatus != WGPUBufferMapAsyncStatus.Success) {
            error("mapAsync() failed: status $wgpuStatus")
        }
    }

    actual fun unmap() = WebGPUInstance.wgpuNative.wgpuBufferUnmap(wgpuBuffer)
}
