package io.kinference

import okio.Buffer

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

expect fun Buffer.writeDouble(value: Double): Buffer
expect fun Buffer.writeDoubleLe(value: Double): Buffer

expect fun Buffer.readDouble(): Double
expect fun Buffer.readDoubleLe(): Double

expect fun Buffer.writeFloat(value: Float): Buffer
expect fun Buffer.writeFloatLe(value: Float): Buffer

expect fun Buffer.readFloat(): Float
expect fun Buffer.readFloatLe(): Float
