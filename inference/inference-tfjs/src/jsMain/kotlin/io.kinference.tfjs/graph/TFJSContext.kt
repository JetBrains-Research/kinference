package io.kinference.tfjs.graph

import io.kinference.graph.Context
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor

class TFJSContext(base: TFJSContext? = null) : Context<TFJSData<*>>(base) {
    override fun removeValues(predicate: (String) -> Boolean) {
        val allToRemove = values.entries.filter { predicate(it.key) }
        allToRemove.forEach {
            if (it.value is TFJSTensor) {
                (it.value as TFJSTensor).data.dispose()
            }
        }
        values.entries.removeAll(allToRemove)
    }
}
