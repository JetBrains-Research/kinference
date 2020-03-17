package org.jetbrains.research.kotlin.mpp.inference.graph

import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

@Suppress("UNCHECKED_CAST")
class NodeIO(names: List<String> = emptyList()) : LinkedHashMap<String, Tensor<*>?>() {
    init {
        for (name in names) this[name] = null
    }

    fun addValue(name: String) {
        this[name] = null
    }

    fun addValues(vals: List<Tensor<*>>): NodeIO {
        vals.forEach { tensor -> this[tensor.name ?: ""] = tensor }
        return this
    }

    fun addNotNullValues(vals: List<Tensor<*>>): NodeIO {
        vals.filter { it.name != null }.forEach { this[it.name!!] = it }
        return this
    }

    fun clearValues(): NodeIO {
        this.keys.forEach { this[it] = null }
        return this
    }

    val availableForWriting: Set<String>
        get() = this.filter { it.value == null }.keys
}
