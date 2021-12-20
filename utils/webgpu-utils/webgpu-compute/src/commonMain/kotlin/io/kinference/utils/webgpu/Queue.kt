package io.kinference.utils.webgpu

import kotlin.math.max

expect class Queue {
    suspend fun onSubmittedWorkDone()
    fun submit(commandBuffers: List<CommandBuffer>)
    fun writeBuffer(
        buffer: Buffer,
        bufferOffset: Int,
        data: BufferData,
        dataOffset: Int = 0,
        size: Int = max(0, data.size - dataOffset)
    )
}
