package io.kinference.utils

interface Closeable {
    suspend fun close()
}

suspend fun <T : Closeable> closeArrays(arrays: Array<T>) = arrays.forEach { it.close() }
suspend fun <T : Closeable> closeAll(arrays: List<T>) = arrays.forEach { it.close() }
suspend fun <T : Closeable> closeAll(vararg array: T?) = array.forEach { it?.close() }
