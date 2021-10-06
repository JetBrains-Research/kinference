package io.kinference.tfjs.utils

fun <T> MutableSet<T>.removeIf(body: (T) -> Boolean) {
    val filtered = this.filter(body)
    this.removeAll(filtered)
}

class Stack<T> {
    private val data = ArrayDeque<T>()

    fun push(value: T) = data.addLast(value)

    fun pop() = data.removeLast()

    fun peek() = data.last()

    fun isEmpty() = data.isEmpty()
    fun isNotEmpty() = data.isNotEmpty()
}
