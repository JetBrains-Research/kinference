package io.kinference.utils.webgpu

import org.khronos.webgl.Uint32Array

actual class ComputePassEncoder(val gpuComputePassEncoder: GPUComputePassEncoder) {
    actual fun dispatch(x: Int, y: Int, z: Int) = gpuComputePassEncoder.dispatch(x, y, z)

    actual fun dispatchIndirect(indirectBuffer: Buffer, indirectOffset: Int) =
        gpuComputePassEncoder.dispatchIndirect(indirectBuffer.gpuBuffer, indirectOffset)

    actual fun endPass() = gpuComputePassEncoder.endPass()

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsets: List<BufferDynamicOffset>) =
        gpuComputePassEncoder.setBindGroup(index, bindGroup, dynamicOffsets.toTypedArray())

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsetsData: BufferData, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int) =
        gpuComputePassEncoder.setBindGroup(index, bindGroup, Uint32Array(dynamicOffsetsData.buffer), dynamicOffsetsDataStart, dynamicOffsetsDataLength)

    actual fun setPipeline(pipeline: ComputePipeline) = gpuComputePassEncoder.setPipeline(pipeline)
}

external class GPUComputePassEncoder {
    fun dispatch(x: Int, y: Int, z: Int)
    fun dispatchIndirect(indirectBuffer: GPUBuffer, indirectOffset: Int)
    fun endPass()
    fun setBindGroup(index: Int, bindGroup: GPUBindGroup, dynamicOffsets: Array<BufferDynamicOffset>)
    fun setBindGroup(index: Int, bindGroup: GPUBindGroup, dynamicOffsetsData: Uint32Array, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int)
    fun setPipeline(pipeline: GPUComputePipeline)
}
