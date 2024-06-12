package io.kinference.utils

import java.util.concurrent.ConcurrentLinkedQueue

actual class ConcurrentQueue<T> actual constructor() {
    private val queue: ConcurrentLinkedQueue<T> = ConcurrentLinkedQueue()

    actual fun removeFirstOrNull(): T? {
        return queue.poll()
    }

    actual fun addLast(element: T) {
        queue.offer(element)
    }

    actual fun clear() {
        queue.clear()
    }
}
