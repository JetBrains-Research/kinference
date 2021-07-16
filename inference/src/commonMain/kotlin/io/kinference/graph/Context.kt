package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.utils.removeIf

class Context(private val base: Context? = null) {
    private val values = HashMap<String, ONNXData>()
    private val shapes = HashMap<String, Int>()

    fun hasValue(name: String): Boolean {
        return values.contains(name) && (base?.hasValue(name) ?: true)
    }

    fun hasShape(name: String): Boolean {
        return shapes.contains(name) && (base?.hasShape(name) ?: true)
    }

    fun putValue(name: String, value: ONNXData) {
        require(name !in values && base?.hasValue(name)?.not() ?: true) { "'$name' already exists in context values" }
        values[name] = value
    }

    fun getValue(name: String): ONNXData {
        return values[name] ?: base?.getValue(name) ?: error("'$name' not found in context values")
    }

    fun getOrNullValue(name: String): ONNXData? {
        return values[name] ?: base?.getOrNullValue(name)
    }

    fun removeValues(predicate: (String) -> Boolean) {
        values.entries.removeIf { predicate(it.key) }
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

    fun mergeContext(context: Context) {
        values.putAll(context.values)
        shapes.putAll(context.shapes)
    }
}
