package io.kinference.utils.webgpu

import kotlinx.coroutines.await
import org.khronos.webgl.BufferDataSource
import kotlin.js.Promise

actual class Queue(private val gpuQueue: GPUQueue) {
    actual suspend fun onSubmittedWorkDone() {
        gpuQueue.onSubmittedWorkDone().await()
    }

    actual fun submit(commandBuffers: List<CommandBuffer>) = gpuQueue.submit(commandBuffers.toTypedArray())

    actual fun writeBuffer(buffer: Buffer, bufferOffset: Int, data: BufferData, dataOffset: Int, size: Int) {
        gpuQueue.writeBuffer(buffer.gpuBuffer, bufferOffset, data.buffer, dataOffset, size)
    }
}

external class GPUQueue {
    fun onSubmittedWorkDone(): Promise<Any?>
    fun submit(commandBuffers: Array<CommandBuffer>)
    fun writeBuffer(buffer: GPUBuffer, bufferOffset: Int, data: BufferDataSource, dataOffset: Int, size: Int)
}
