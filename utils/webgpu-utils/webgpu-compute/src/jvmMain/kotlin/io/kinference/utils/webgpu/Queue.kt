package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.WgpuRuntime
import io.kinference.utils.wgpu.jnr.WGPUQueue
import jnr.ffi.Pointer

actual class Queue(private val wgpuQueue: WGPUQueue) {
    actual suspend fun onSubmittedWorkDone() {
        // TODO
    }

    actual fun submit(commandBuffers: List<CommandBuffer>) {
        val wgpuCommandBuffers = commandBuffers.map { it.wgpuCommandBuffer.address() }.toLongArray().createPointerTo()
        WebGPUInstance.wgpuNative.wgpuQueueSubmit(wgpuQueue, commandBuffers.size.toLong(), wgpuCommandBuffers)
    }

    actual fun writeBuffer(buffer: Buffer, bufferOffset: Int, data: BufferData, dataOffset: Int, size: Int) {
        WebGPUInstance.wgpuNative.wgpuQueueWriteBuffer(
            queue = wgpuQueue,
            buffer = buffer.wgpuBuffer,
            bufferOffset = bufferOffset.toLong(),
            data = Pointer.wrap(WgpuRuntime.runtime, data.pointer.address() + dataOffset, size.toLong()),
            size = size.toLong()
        )
    }
}
