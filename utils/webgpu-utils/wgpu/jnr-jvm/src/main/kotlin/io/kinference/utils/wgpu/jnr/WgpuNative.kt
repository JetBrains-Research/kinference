package io.kinference.utils.wgpu.jnr

import jnr.ffi.Pointer
import jnr.ffi.annotations.In
import jnr.ffi.annotations.Out
import jnr.ffi.types.*

interface WgpuNative {
    // webgpu.h

    // Methods of Adapter
    fun wgpuAdapterGetLimits(@In adapter: WGPUAdapter, @Out limits: WGPUSupportedLimitsPointer): Boolean
    fun wgpuAdapterRequestDevice(@In adapter: WGPUAdapter, @In descriptor: WGPUDeviceDescriptorPointer, callback: WGPURequestDeviceCallback, userdata: Pointer)

    // Methods of Buffer
    fun wgpuBufferDestroy(@In buffer: WGPUBuffer)
    fun wgpuBufferGetMappedRange(@In buffer: WGPUBuffer, @size_t offset: Long, @size_t size: Long): Pointer
    fun wgpuBufferMapAsync(@In buffer: WGPUBuffer, @u_int32_t mode: WGPUMapModeFlags, @size_t offset: Long, @size_t size: Long, callback: WGPUBufferMapCallback, userdata: Pointer)
    fun wgpuBufferUnmap(@In buffer: WGPUBuffer)

    // Methods of CommandEncoder
    fun wgpuCommandEncoderBeginComputePass(@In commandEncoder: WGPUCommandEncoder, @In descriptor: WGPUComputePassDescriptorPointer): WGPUComputePassEncoder
    fun wgpuCommandEncoderCopyBufferToBuffer(@In commandEncoder: WGPUCommandEncoder, @In source: WGPUBuffer, @u_int64_t sourceOffset: Long, destination: WGPUBuffer, @u_int64_t destinationOffset: Long, @u_int64_t size: Long)
    fun wgpuCommandEncoderFinish(@In commandEncoder: WGPUCommandEncoder, @In descriptor: WGPUCommandBufferDescriptor): WGPUCommandBuffer

    // Methods of ComputePassEncoder
    fun wgpuComputePassEncoderDispatch(@In computePassEncoder: WGPUComputePassEncoder, @u_int32_t x: Long, @u_int32_t y: Long, @u_int32_t z: Long)
    fun wgpuComputePassEncoderDispatchIndirect(@In computePassEncoder: WGPUComputePassEncoder, @In indirectBuffer: WGPUBuffer, @u_int64_t indirectOffset: Long)
    fun wgpuComputePassEncoderEndPass(@In computePassEncoder: WGPUComputePassEncoder)
    fun wgpuComputePassEncoderSetBindGroup(@In computePassEncoder: WGPUComputePassEncoder, @u_int32_t groupIndex: Long, @In group: WGPUBindGroup, @u_int32_t dynamicOffsetCount: Long, @In dynamicOffsets: Pointer)
    fun wgpuComputePassEncoderSetPipeline(@In computePassEncoder: WGPUComputePassEncoder, @In pipeline: WGPUComputePipeline)

    // Methods of ComputePipeline
    fun wgpuComputePipelineGetBindGroupLayout(@In computePipeline: WGPUComputePipeline, @u_int32_t groupIndex: Long): WGPUBindGroupLayout

    // Methods of Device
    fun wgpuDeviceCreateBindGroup(@In device: WGPUDevice, @In descriptor: WGPUBindGroupDescriptorPointer): WGPUBindGroup
    fun wgpuDeviceCreateBindGroupLayout(@In device: WGPUDevice, @In descriptor: WGPUBindGroupLayoutDescriptorPointer): WGPUBindGroupLayout
    fun wgpuDeviceCreateBuffer(@In device: WGPUDevice, @In descriptor: WGPUBufferDescriptorPointer): WGPUBuffer
    fun wgpuDeviceCreateCommandEncoder(@In device: WGPUDevice, @In descriptor: WGPUCommandBufferDescriptorPointer): WGPUCommandEncoder
    fun wgpuDeviceCreateComputePipeline(@In device: WGPUDevice, @In descriptor: WGPUComputePipelineDescriptorPointer): WGPUComputePipeline
    fun wgpuDeviceCreatePipelineLayout(@In device: WGPUDevice, @In descriptor: WGPUPipelineLayoutDescriptorPointer): WGPUPipelineLayout
    fun wgpuDeviceCreateShaderModule(@In device: WGPUDevice, @In descriptor: WGPUShaderModuleDescriptorPointer): WGPUShaderModule
    fun wgpuDeviceDestroy(@In device: WGPUDevice)
    fun wgpuDeviceGetLimits(@In device: WGPUDevice, @Out limits: WGPUSupportedLimitsPointer): Boolean
    fun wgpuDeviceGetQueue(@In device: WGPUDevice): WGPUQueue

    // Methods of Instance
    fun wgpuInstanceRequestAdapter(@In instance: WGPUInstance, @In options: WGPURequestAdapterOptionsPointer, callback: WGPURequestAdapterCallback, userdata: Pointer)

    // Methods of Queue
    fun wgpuQueueOnSubmittedWorkDone(@In queue: WGPUQueue, @u_int64_t signalValue: Long, callback: WGPUQueueWorkDoneCallback, userdata: Pointer)
    fun wgpuQueueSubmit(@In queue: WGPUQueue, @u_int32_t commandCount: Long, @In commands: Pointer)
    fun wgpuQueueWriteBuffer(@In queue: WGPUQueue, @In buffer: WGPUBuffer, @u_int64_t bufferOffset: Long, @In data: Pointer, @size_t size: Long)


    // wgpu.h

    fun wgpuDevicePoll(@In device: WGPUDevice, force_wait: Boolean)
}
