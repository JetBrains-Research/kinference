package io.kinference.graph

class GraphValueOrderInfo {
    // By storing IntArray with 1 element instead of Int
    // we prevent boxing operations in getOrder() function
    private val orders: LinkedHashMap<String, IntArray> = linkedMapOf()

    fun putOrder(name: String, order: Int) {
        if (!orders.containsKey(name))
            orders[name] = intArrayOf(order)
        else if (orders[name]!![0] < order)
            orders[name]!![0] = order
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
        return orders[name]?.get(0) ?: Int.MAX_VALUE
    }

    fun names(): Set<String> {
        return orders.keys
    }
}
