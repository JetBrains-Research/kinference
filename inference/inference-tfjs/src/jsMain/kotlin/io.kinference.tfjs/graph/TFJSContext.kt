package io.kinference.tfjs.graph

import io.kinference.graph.GraphContext
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor

class TFJSGraphContext(base: TFJSGraphContext? = null) : GraphContext<TFJSData<*>>(base) {
    override fun removeValues(predicate: (String) -> Boolean) {
        val allToRemove = values.entries.filter { predicate(it.key) }
        allToRemove.forEach { it.value.close() }
        values.entries.removeAll(allToRemove)
    }
}
