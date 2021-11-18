package io.kinference.core.graph

import io.kinference.core.KIONNXData
import io.kinference.utils.removeIf
import io.kinference.graph.Context

class KIContext(base: KIContext? = null) : Context<KIONNXData<*>>(base) {
    override fun removeValues(predicate: (String) -> Boolean) {
        values.entries.removeIf { predicate(it.key) }
    }
}
