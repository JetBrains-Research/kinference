package io.kinference.utils

actual typealias Closeable = java.io.Closeable

actual fun Throwable.addSuppressedException(other: Throwable) {
    this.addSuppressed(other)
}
