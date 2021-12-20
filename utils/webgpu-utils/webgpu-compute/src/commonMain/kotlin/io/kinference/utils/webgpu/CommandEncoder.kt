package io.kinference.utils.webgpu

expect class CommandEncoder {
    fun beginComputePass(descriptor: ComputePassDescriptor = ComputePassDescriptor()): ComputePassEncoder
    fun copyBufferToBuffer(source: Buffer, sourceOffset: Int, destination: Buffer, destinationOffset: Int, size: Int)
    fun finish(descriptor: CommandBufferDescriptor = CommandBufferDescriptor()): CommandBuffer
}
