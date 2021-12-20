package io.kinference.utils.webgpu

actual class CommandEncoder(val gpuCommandEncoder: GPUCommandEncoder) {
    actual fun beginComputePass(descriptor: ComputePassDescriptor): ComputePassEncoder =
        ComputePassEncoder(gpuCommandEncoder.beginComputePass(descriptor))

    actual fun copyBufferToBuffer(source: Buffer, sourceOffset: Int, destination: Buffer, destinationOffset: Int, size: Int) =
        gpuCommandEncoder.copyBufferToBuffer(source.gpuBuffer, sourceOffset, destination.gpuBuffer, destinationOffset, size)

    actual fun finish(descriptor: CommandBufferDescriptor): CommandBuffer =
        gpuCommandEncoder.finish(descriptor)
}

external class GPUCommandEncoder {
    fun beginComputePass(descriptor: ComputePassDescriptor): GPUComputePassEncoder
    fun copyBufferToBuffer(source: GPUBuffer, sourceOffset: Int, destination: GPUBuffer, destinationOffset: Int, size: Int)
    fun finish(descriptor: CommandBufferDescriptor): GPUCommandBuffer
}
