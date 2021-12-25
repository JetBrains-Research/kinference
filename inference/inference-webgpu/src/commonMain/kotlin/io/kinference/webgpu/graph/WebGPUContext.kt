package io.kinference.webgpu.graph

import io.kinference.graph.Context
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData

class WebGPUContext(
    val gpuState: WebGPUState,
    base: Context<WebGPUData<*>>? = null
) : Context<WebGPUData<*>>(base) {
    private val valuesToDestroy: MutableList<WebGPUData<*>> = arrayListOf()

    override fun removeValues(predicate: (String) -> Boolean) {
        val allToRemove = values.entries.filter { predicate(it.key) }
        valuesToDestroy.addAll(allToRemove.map { it.value })
        values.entries.removeAll(allToRemove)
    }

    fun destroyRemovedValues() {
        valuesToDestroy.forEach {
            if (it is WebGPUTensor) {
                it.data.destroy()
            }
        }
        valuesToDestroy.clear()
    }
}
