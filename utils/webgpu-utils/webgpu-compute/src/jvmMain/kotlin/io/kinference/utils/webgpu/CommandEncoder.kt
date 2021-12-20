package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.getPointerTo
import io.kinference.utils.wgpu.jnr.WGPUCommandEncoder

actual class CommandEncoder(private val wgpuCommandEncoder: WGPUCommandEncoder) {
    actual fun beginComputePass(descriptor: ComputePassDescriptor): ComputePassEncoder =
        ComputePassEncoder(
            WebGPUInstance.wgpuNative.wgpuCommandEncoderBeginComputePass(
                wgpuCommandEncoder,
                descriptor.getPointerTo()
            )
        )

    actual fun copyBufferToBuffer(source: io.kinference.utils.webgpu.Buffer, sourceOffset: Int, destination: io.kinference.utils.webgpu.Buffer, destinationOffset: Int, size: Int) =
        WebGPUInstance.wgpuNative.wgpuCommandEncoderCopyBufferToBuffer(
            wgpuCommandEncoder, source.wgpuBuffer, sourceOffset.toLong(), destination.wgpuBuffer, destinationOffset.toLong(), size.toLong()
        )

    actual fun finish(descriptor: CommandBufferDescriptor): CommandBuffer =
        CommandBuffer(WebGPUInstance.wgpuNative.wgpuCommandEncoderFinish(wgpuCommandEncoder, descriptor))
}
