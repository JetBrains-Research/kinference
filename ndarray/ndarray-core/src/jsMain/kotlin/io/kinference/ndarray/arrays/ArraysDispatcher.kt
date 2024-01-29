package io.kinference.ndarray.arrays

import kotlinx.coroutines.Dispatchers
import kotlin.coroutines.CoroutineContext

internal actual fun getArraysDispatcherContext(): CoroutineContext = Dispatchers.Default

actual inline fun <reified T> ArraysDispatcher.getArrays(
    type: ArrayTypes,
    size: Int,
    count: Int
): Array<T> {
    return Array(count) { (getArray(type, size)).array as T }
}

actual inline fun <reified T> ArraysDispatcher.getArraysAndMarkers(
    type: ArrayTypes,
    size: Int,
    count: Int
): Array<ArrayContainer<T>> {
    return Array(count) { getArray(type, size) as ArrayContainer<T> }
}
