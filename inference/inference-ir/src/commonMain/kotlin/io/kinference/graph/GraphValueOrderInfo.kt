package io.kinference.graph

class GraphValueOrderInfo {
    private val orders: HashMap<String, Int> = HashMap()

    fun putOrder(name: String, order: Int) {
        if (!orders.containsKey(name) || orders[name]!! < order)
            orders[name] = order
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
        return orders.getOrElse(name) { Int.MAX_VALUE }
    }

    fun names(): Set<String> {
        return orders.keys
    }
}
