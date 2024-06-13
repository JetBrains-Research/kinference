package io.kinference.utils

actual class ConcurrentQueue<T> actual constructor() {
    private val queue: ArrayDeque<T> = ArrayDeque()

    actual fun removeFirstOrNull(): T? {
        return queue.removeFirstOrNull()
    }

    actual fun addLast(element: T) {
        queue.addLast(element)
    }

    actual fun clear() {
        queue.clear()
    }
}
