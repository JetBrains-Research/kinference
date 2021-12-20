package io.kinference.utils.webgpu

expect object WebGPUInstance {
    suspend fun requestAdapter(options: RequestAdapterOptions = RequestAdapterOptions()): Adapter
}
