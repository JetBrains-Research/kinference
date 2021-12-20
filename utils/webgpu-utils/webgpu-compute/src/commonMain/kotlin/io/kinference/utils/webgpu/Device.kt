package io.kinference.utils.webgpu

expect class Device {
    val limits: SupportedLimits
    val queue: Queue

    fun createBindGroup(descriptor: BindGroupDescriptor): BindGroup
    fun createBindGroupLayout(descriptor: BindGroupLayoutDescriptor): BindGroupLayout
    fun createBuffer(descriptor: BufferDescriptor): Buffer
    fun createCommandEncoder(descriptor: CommandEncoderDescriptor = CommandEncoderDescriptor()): CommandEncoder
    fun createComputePipeline(descriptor: ComputePipelineDescriptor): ComputePipeline
    fun createPipelineLayout(descriptor: PipelineLayoutDescriptor): PipelineLayout
    fun createShaderModule(descriptor: ShaderModuleDescriptor): ShaderModule
    fun destroy()
}
