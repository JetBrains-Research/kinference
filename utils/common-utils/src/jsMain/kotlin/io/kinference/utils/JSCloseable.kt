package io.kinference.utils

actual interface Closeable {
    actual fun close()
}

actual fun Throwable.addSuppressedException(other: Throwable) = Unit
