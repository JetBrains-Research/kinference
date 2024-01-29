package io.kinference.ndarray.arrays

import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.runBlocking
import kotlin.coroutines.CoroutineContext

@OptIn(DelicateCoroutinesApi::class)
internal actual fun getArraysDispatcherContext(): CoroutineContext = newSingleThreadContext("ArraysDispatcherContext")

actual inline fun <reified T> ArraysDispatcher.getArrays(
    type: ArrayTypes,
    size: Int,
    count: Int
): Array<T> = runBlocking(singleThreadContext) {
    Array(count) { (getArray(type, size)).array as T }
}

actual inline fun <reified T> ArraysDispatcher.getArraysAndMarkers(
    type: ArrayTypes,
    size: Int,
    count: Int
): Array<ArrayContainer<T>> = runBlocking(singleThreadContext) {
    Array(count) { getArray(type, size) as ArrayContainer<T> }
}
