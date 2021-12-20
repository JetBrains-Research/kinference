package io.kinference.utils.webgpu

import kotlinx.coroutines.await
import org.khronos.webgl.ArrayBuffer
import kotlin.js.Promise

actual class Buffer(val gpuBuffer: GPUBuffer, actual val size: Int) {
    actual fun destroy() = gpuBuffer.destroy()

    actual fun getMappedRange(offset: Int, size: Int): BufferData = BufferData(gpuBuffer.getMappedRange(offset, size))

    actual suspend fun mapAsync(mode: MapModeFlags, offset: Int, size: Int) {
        val actualSize = if (size >= 0) size else maxOf(0, this.size - offset)
        gpuBuffer.mapAsync(mode.value, offset, actualSize).await()
    }

    actual fun unmap() = gpuBuffer.unmap()
}

external class GPUBuffer {
    fun destroy()
    fun getMappedRange(offset: Int, size: Int): ArrayBuffer
    fun mapAsync(mode: GPUMapModeFlags, offset: Int, size: Int): Promise<Any?>
    fun unmap()
}
