package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.jnr.WGPUComputePassEncoder

actual class ComputePassEncoder(private val wgpuComputePassEncoder: WGPUComputePassEncoder) {
    actual fun dispatch(x: Int, y: Int, z: Int) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderDispatch(
            wgpuComputePassEncoder, x.toLong(), y.toLong(), z.toLong()
        )

    actual fun dispatchIndirect(indirectBuffer: Buffer, indirectOffset: Int) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderDispatchIndirect(
            wgpuComputePassEncoder, indirectBuffer.wgpuBuffer, indirectOffset.toLong()
        )

    actual fun endPass() = WebGPUInstance.wgpuNative.wgpuComputePassEncoderEndPass(wgpuComputePassEncoder)

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsets: List<BufferDynamicOffset>) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderSetBindGroup(
            wgpuComputePassEncoder, index.toLong(), bindGroup.wgpuBindGroup, dynamicOffsets.size.toLong(),
            dynamicOffsets.toLongArray().createPointerTo()
        )

    actual fun setBindGroup(
        index: Int, bindGroup: BindGroup, dynamicOffsetsData: BufferData,
        dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int
    ) {
        TODO("Not yet implemented")
    }

    actual fun setPipeline(pipeline: ComputePipeline) = WebGPUInstance.wgpuNative.wgpuComputePassEncoderSetPipeline(
        wgpuComputePassEncoder, pipeline.wgpuComputePipeline
    )
}
