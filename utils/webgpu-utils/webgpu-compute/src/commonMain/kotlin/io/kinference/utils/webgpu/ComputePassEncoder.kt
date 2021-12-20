package io.kinference.utils.webgpu

expect class ComputePassEncoder {
    fun dispatch(x: Int, y: Int = 1, z: Int = 1)
    fun dispatchIndirect(indirectBuffer: Buffer, indirectOffset: Int)
    fun endPass()
    fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsets: List<BufferDynamicOffset>)
    fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsetsData: BufferData, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int)
    fun setPipeline(pipeline: ComputePipeline)
}
