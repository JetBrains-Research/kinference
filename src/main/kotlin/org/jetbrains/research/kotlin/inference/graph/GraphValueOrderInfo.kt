package org.jetbrains.research.kotlin.inference.graph

class GraphValueOrderInfo {
    private val orders: HashMap<String, Int> = HashMap()

    fun putOrder(name: String, order: Int) {
        if (!orders.containsKey(name) || orders[name]!! < order)
            orders[name] = order
    }

    fun putOrder(names: Set<String>, order: Int) {
        for (name in names) {
            putOrder(name, order)
        }
    }

    fun getOrder(name: String): Int {
        return orders.getOrDefault(name, Int.MAX_VALUE)
    }
}
