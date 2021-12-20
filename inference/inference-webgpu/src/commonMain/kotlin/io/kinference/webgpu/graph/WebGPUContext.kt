package io.kinference.webgpu.graph

import io.kinference.graph.Context
import io.kinference.utils.webgpu.CommandEncoder
import io.kinference.utils.webgpu.Device
import io.kinference.webgpu.engine.WebGPUData

class WebGPUContext(
    internal val device: Device,
    internal val commandEncoder: CommandEncoder = device.createCommandEncoder(),
    base: Context<WebGPUData<*>>? = null
) : Context<WebGPUData<*>>(base) {
    override fun removeValues(predicate: (String) -> Boolean) {
        val allToRemove = values.entries.filter { predicate(it.key) }
        /*
        allToRemove.forEach {
            if (it.value is WebGPUTensor) {
                (it.value as WebGPUTensor).data.buffer.destroy()
            }
        }
         */
        values.entries.removeAll(allToRemove)
    }
}
