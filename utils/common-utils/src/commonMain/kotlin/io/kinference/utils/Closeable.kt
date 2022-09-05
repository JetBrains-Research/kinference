package io.kinference.utils

expect interface Closeable {
    fun close()
}

expect fun Throwable.addSuppressedException(other: Throwable)

inline fun <C : Closeable, T> C.use(func: (C) -> T): T {
    var isClosed = false

    return try {
        func(this)
    } catch (first: Throwable) {
        try {
            isClosed = true
            close()
        } catch (second: Throwable) {
            first.addSuppressedException(second)
        }
        throw first
    } finally {
        if (!isClosed) close()
    }
}
