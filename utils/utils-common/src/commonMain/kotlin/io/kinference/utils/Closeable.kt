package io.kinference.utils

interface Closeable {
    fun close()
}

 fun <T : Closeable> closeArrays(arrays: Array<T>) = arrays.forEach { it.close() }
 fun <T : Closeable> closeAll(arrays: List<T>) = arrays.forEach { it.close() }
 fun <T : Closeable> closeAll(vararg array: T?) = array.forEach { it?.close() }
