package io.kinference.utils

actual interface Closeable {
    actual suspend fun close()
}
