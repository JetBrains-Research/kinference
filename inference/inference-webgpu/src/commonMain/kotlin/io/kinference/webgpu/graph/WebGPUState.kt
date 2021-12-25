package io.kinference.webgpu.graph

import io.kinference.utils.webgpu.*
import io.kinference.webgpu.ndarray.*

class WebGPUState(val device: Device) {
    private var numOfPendingCommands: Int = 0
    private var commandEncoder: CommandEncoder = device.createCommandEncoder()

    fun beginComputePass(descriptor: ComputePassDescriptor = ComputePassDescriptor()): ComputePassEncoder {
        numOfPendingCommands++
        return commandEncoder.beginComputePass(descriptor)
    }

    fun copyBufferToBuffer(source: Buffer, sourceOffset: Int, destination: Buffer, destinationOffset: Int, size: Int) {
        numOfPendingCommands++
        commandEncoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)
    }

    fun enqueueCommands() {
        if (numOfPendingCommands > 0) {
            val commandBuffer = commandEncoder.finish()
            device.queue.submit(listOf(commandBuffer))
            commandEncoder = device.createCommandEncoder()
            numOfPendingCommands = 0
        }
    }

    fun createInitializedBuffer(info: NDArrayInfo, bufferData: TypedNDArrayData): Buffer {
        val buffer = device.createBuffer(
            BufferDescriptor(
                size = info.sizeBytes,
                usage = BufferUsageFlags(BufferUsage.Storage, BufferUsage.CopySrc),
                mappedAtCreation = true
            )
        )
        when (bufferData) {
            is IntNDArrayData -> buffer.getMappedRange().set(bufferData.data)
            is UIntNDArrayData -> buffer.getMappedRange().set(bufferData.data.asIntArray())
            is FloatNDArrayData -> buffer.getMappedRange().set(bufferData.data)
        }
        buffer.unmap()
        return buffer
    }

    fun createUninitializedBuffer(info: NDArrayInfo): Buffer =
        device.createBuffer(
            BufferDescriptor(
                size = info.sizeBytes,
                usage = BufferUsageFlags(BufferUsage.Storage, BufferUsage.CopySrc, BufferUsage.CopyDst),
                mappedAtCreation = false
            )
        )

    fun createReadableBuffer(info: NDArrayInfo, sourceBuffer: Buffer): Buffer {
        val destinationBuffer = device.createBuffer(
            BufferDescriptor(
                size = info.sizeBytes,
                usage = BufferUsageFlags(BufferUsage.MapRead, BufferUsage.CopyDst),
                mappedAtCreation = false
            )
        )
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, destinationBuffer, 0, info.sizeBytes)
        return destinationBuffer
    }
}
