package io.kinference.utils.webgpu

actual class Device(private val gpuDevice: GPUDevice) {
    actual val limits
        get() = gpuDevice.limits

    actual val queue: Queue
        get() = Queue(gpuDevice.queue)

    actual fun createBindGroup(descriptor: BindGroupDescriptor): BindGroup = gpuDevice.createBindGroup(descriptor)

    actual fun createBindGroupLayout(descriptor: BindGroupLayoutDescriptor): BindGroupLayout =
        gpuDevice.createBindGroupLayout(descriptor)

    actual fun createBuffer(descriptor: BufferDescriptor): Buffer =
        Buffer(gpuDevice.createBuffer(descriptor), descriptor.size)

    actual fun createCommandEncoder(descriptor: CommandEncoderDescriptor): CommandEncoder =
        CommandEncoder(gpuDevice.createCommandEncoder(descriptor))

    actual fun createComputePipeline(descriptor: ComputePipelineDescriptor): ComputePipeline =
        gpuDevice.createComputePipeline(descriptor)

    actual fun createPipelineLayout(descriptor: PipelineLayoutDescriptor): PipelineLayout =
        gpuDevice.createPipelineLayout(descriptor)

    actual fun createShaderModule(descriptor: ShaderModuleDescriptor): ShaderModule =
        ShaderModule(gpuDevice.createShaderModule(descriptor))

    actual fun destroy() = gpuDevice.destroy()

    actual suspend fun wait() {
        queue.onSubmittedWorkDone()
    }
}

external class GPUDevice {
    val limits: GPUSupportedLimits
    val queue: GPUQueue

    fun createBindGroup(descriptor: BindGroupDescriptor): GPUBindGroup
    fun createBindGroupLayout(descriptor: BindGroupLayoutDescriptor): GPUBindGroupLayout
    fun createBuffer(descriptor: BufferDescriptor): GPUBuffer
    fun createCommandEncoder(descriptor: CommandEncoderDescriptor): GPUCommandEncoder
    fun createComputePipeline(descriptor: ComputePipelineDescriptor): GPUComputePipeline
    fun createPipelineLayout(descriptor: PipelineLayoutDescriptor): GPUPipelineLayout
    fun createShaderModule(descriptor: ShaderModuleDescriptor): GPUShaderModule
    fun destroy()
}
