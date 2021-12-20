package io.kinference.webgpu.ndarray

import io.kinference.utils.webgpu.*

class NDArray(val info: ArrayInfo, data: Any? = null, isOutput: Boolean = false, device: Device) {
    internal val buffer: Buffer
    private var mappedRange: BufferData? = null

    suspend fun map(flags: MapModeFlags) {
        if (mappedRange == null) {
            buffer.mapAsync(flags)
            mappedRange = buffer.getMappedRange()
        }
    }

    fun unmap() {
        if (mappedRange != null) {
            buffer.unmap()
            mappedRange = null
        }
    }

    fun getMappedRange(): BufferData = mappedRange!!

    init {
        val usage = if (isOutput) {
            BufferUsageFlags(BufferUsage.MapRead, BufferUsage.CopyDst)
        } else {
            BufferUsageFlags(BufferUsage.Storage, BufferUsage.CopySrc, BufferUsage.CopyDst)
        }
        buffer = device.createBuffer(
            BufferDescriptor(
                size = info.sizeBytes,
                usage = usage,
                mappedAtCreation = data != null
            )
        )
        if (data != null) {
            mappedRange = buffer.getMappedRange()
            when (data) {
                is IntArray -> getMappedRange().set(data)
                is FloatArray -> getMappedRange().set(data)
                is BufferData -> getMappedRange().set(data)
                else -> error("Unknown data type")
            }
        }
    }

    fun destroy() {
        buffer.destroy()
    }
}
