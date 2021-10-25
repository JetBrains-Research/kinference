package io.kinference.core.graph

import io.kinference.core.KIONNXData
import io.kinference.core.utils.removeIf
import io.kinference.data.ONNXData

class Context(private val base: Context? = null) {
    private val values = HashMap<String, KIONNXData<*>>()
    private val shapes = HashMap<String, Int>()

    fun hasValue(name: String): Boolean {
        return values.contains(name) && (base?.hasValue(name) ?: true)
    }

    fun hasShape(name: String): Boolean {
        return shapes.contains(name) && (base?.hasShape(name) ?: true)
    }

    fun putValue(name: String, value: KIONNXData<*>) {
        require(name !in values && base?.hasValue(name)?.not() ?: true) { "'$name' already exists in context values" }
        values[name] = value
    }

    fun getValue(name: String): KIONNXData<*> {
        return values[name] ?: base?.getValue(name) ?: error("'$name' not found in context values")
    }

    fun getOrNullValue(name: String): KIONNXData<*>? {
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
