package org.jetbrains.research.kotlin.mpp.inference.graph

import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor

class Context(val base: Context? = null) {
    private val values = HashMap<String, Tensor>()
    private val shapes = HashMap<String, Int>()

    fun hasValue(name: String): Boolean {
        return values.contains(name)
    }

    fun hasShape(name: String): Boolean {
        return shapes.contains(name)
    }

    fun putValue(name: String, value: Tensor) {
        require(name !in values && base?.hasValue(name)?.not() ?: true) { "'$name' already exists in context values" }
        values[name] = value
    }

    fun getValue(name: String): Tensor {
        return values[name] ?: base?.getValue(name) ?: error("'$name' not found in context values")
    }

    fun putShape(name: String, shape: Int) {
        require(name !in shapes && base?.hasShape(name)?.not() ?: true) { "'$name' already exists in context shapes" }
        shapes[name] = shape
    }

    fun getShape(name: String): Int {
        return shapes[name] ?: base?.getShape(name) ?: error("'$name' not found in context shapes")
    }

    fun clear() {
        values.clear()
        shapes.clear()
    }
}
