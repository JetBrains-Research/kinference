package org.jetbrains.research.kotlin.mpp.inference.graph

import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class NodeIO(names: List<String> = emptyList()) {
    private val namedTensors: LinkedHashMap<String, Tensor<*>?> = LinkedHashMap()

    init {
        for (name in names) addName(name)
    }

    val names: Set<String>
        get() = namedTensors.keys

    val tensors: MutableCollection<Tensor<*>?>
        get() = namedTensors.values

    val availableForWriting: Set<String>
        get() = namedTensors.filter { it.value == null }.keys

    operator fun get(name: String): Tensor<*>? {
        return namedTensors[name]
    }

    operator fun set(name: String, value: Tensor<*>?) {
        namedTensors[name] = value
    }

    fun addName(name: String): NodeIO {
        namedTensors[name] = null
        return this
    }

    fun addTensors(tensors: List<Tensor<*>>): NodeIO {
        val namedInputTensors = tensors.filter { it.name != null }
        for (tensor in namedInputTensors){
            namedTensors[tensor.name!!] = tensor
        }
        return this
    }

    fun addNamedTensors(tensors: Map<String, Tensor<*>?>): NodeIO {
        namedTensors.putAll(tensors)
        return this
    }

    fun merge(io: NodeIO): NodeIO {
        namedTensors.putAll(io.namedTensors)
        return this
    }

    fun clearValues(): NodeIO {
        namedTensors.keys.forEach { namedTensors[it] = null }
        return this
    }

    fun filterNames(predicate: (String) -> Boolean): Map<String, Tensor<*>?> {
        return namedTensors.filterKeys(predicate)
    }
}
