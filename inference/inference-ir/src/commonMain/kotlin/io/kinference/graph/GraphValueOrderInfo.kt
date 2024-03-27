package io.kinference.graph

import io.kinference.utils.InlineInt

class GraphValueOrderInfo {
    // By storing InlineInt we prevent boxing operations in getOrder() function
    private val orders: LinkedHashMap<String, InlineInt> = linkedMapOf()

    fun putOrder(name: String, order: Int) {
        if (!orders.containsKey(name) || orders[name]!!.value < order)
            orders[name] = InlineInt(order)
    }

    fun putOrder(names: Collection<String>, order: Int) {
        for (name in names) {
            putOrder(name, order)
        }
    }

    fun removeOrder(name: String) {
        if (orders.containsKey(name))
            orders.remove(name)
    }

    fun getOrder(name: String): Int {
        return orders[name]?.value ?: Int.MAX_VALUE
    }

    fun names(): Set<String> {
        return orders.keys
    }
}
