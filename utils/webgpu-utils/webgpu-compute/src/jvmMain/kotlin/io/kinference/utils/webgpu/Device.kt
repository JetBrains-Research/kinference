package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.getPointerTo
import io.kinference.utils.wgpu.jnr.WGPUDevice
import io.kinference.utils.wgpu.jnr.WGPUSupportedLimits

actual class Device(private val wgpuDevice: WGPUDevice) {
    actual val limits: SupportedLimits
        get() {
            val limits = WGPUSupportedLimits.allocateDirect()
            WebGPUInstance.wgpuNative.wgpuDeviceGetLimits(wgpuDevice, limits.getPointerTo())
            return SupportedLimits(limits)
        }

    actual val queue: Queue
        get() = Queue(WebGPUInstance.wgpuNative.wgpuDeviceGetQueue(wgpuDevice))

    actual fun createBindGroup(descriptor: BindGroupDescriptor): BindGroup =
        BindGroup(WebGPUInstance.wgpuNative.wgpuDeviceCreateBindGroup(wgpuDevice, descriptor.getPointerTo()))

    actual fun createBindGroupLayout(descriptor: BindGroupLayoutDescriptor): BindGroupLayout =
        BindGroupLayout(WebGPUInstance.wgpuNative.wgpuDeviceCreateBindGroupLayout(wgpuDevice, descriptor.getPointerTo()))

    actual fun createBuffer(descriptor: BufferDescriptor): Buffer =
        Buffer(
            wgpuBuffer = WebGPUInstance.wgpuNative.wgpuDeviceCreateBuffer(wgpuDevice, descriptor.getPointerTo()),
            size = descriptor.size.toInt(),
            wgpuDevice = wgpuDevice
        )

    actual fun createCommandEncoder(descriptor: CommandEncoderDescriptor): CommandEncoder =
        CommandEncoder(
            WebGPUInstance.wgpuNative.wgpuDeviceCreateCommandEncoder(
                wgpuDevice,
                descriptor.getPointerTo()
            )
        )

    actual fun createComputePipeline(descriptor: ComputePipelineDescriptor): ComputePipeline =
        ComputePipeline(WebGPUInstance.wgpuNative.wgpuDeviceCreateComputePipeline(wgpuDevice, descriptor.getPointerTo()))

    actual fun createPipelineLayout(descriptor: PipelineLayoutDescriptor): PipelineLayout =
        PipelineLayout(WebGPUInstance.wgpuNative.wgpuDeviceCreatePipelineLayout(wgpuDevice, descriptor.getPointerTo()))

    actual fun createShaderModule(descriptor: ShaderModuleDescriptor): ShaderModule =
        ShaderModule(WebGPUInstance.wgpuNative.wgpuDeviceCreateShaderModule(wgpuDevice, descriptor.descriptor.getPointerTo()))

    actual fun destroy() = WebGPUInstance.wgpuNative.wgpuDeviceDestroy(wgpuDevice)

    actual suspend fun wait() {
        WebGPUInstance.wgpuNative.wgpuDevicePoll(wgpuDevice, force_wait = true)
    }
}
