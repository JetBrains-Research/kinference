package io.kinference.webgpu.engine

import io.kinference.utils.webgpu.Device
import io.kinference.utils.webgpu.WebGPUInstance

actual object WebGPUEnvironment {
    private var device: Device? = null

    actual suspend fun getDevice(): Device {
        if (device == null) {
            device = WebGPUInstance.requestAdapter().requestDevice()
        }
        return device!!
    }
}
