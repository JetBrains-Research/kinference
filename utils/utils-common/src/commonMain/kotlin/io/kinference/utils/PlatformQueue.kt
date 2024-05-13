package io.kinference.utils

expect class PlatformQueue<T>() {
    fun removeFirstOrNull(): T?
    fun addLast(element: T)
}
