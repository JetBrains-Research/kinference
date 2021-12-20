package io.kinference.utils.webgpu

expect class Adapter {
    val limits: SupportedLimits

    suspend fun requestDevice(descriptor: DeviceDescriptor = DeviceDescriptor()): Device
}
