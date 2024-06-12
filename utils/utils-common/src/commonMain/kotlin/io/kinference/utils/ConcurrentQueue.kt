package io.kinference.utils

expect class ConcurrentQueue<T>() {
    fun removeFirstOrNull(): T?
    fun addLast(element: T)

    fun clear()
}
