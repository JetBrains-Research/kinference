package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.core.tidy

fun tidy(fn: () -> Array<ArrayTFJS>): Array<ArrayTFJS> = tidy(fn, null)

@Suppress("UNCHECKED_CAST")
fun <T : NDArrayTFJS> tidyNDArrays(fn: () -> Array<T>): Array<T> {
    val rawOutput = tidy {
        val output = fn()
        return@tidy output.map { it.tfjsArray }.toTypedArray()
    }
    return rawOutput.map { it.toNDArray() as T }.toTypedArray()
}

@Suppress("UNCHECKED_CAST")
fun <T : NDArrayTFJS> tidyNDArray(fn: () -> T): T {
    val rawOutput = tidy {
        val output = fn()
        return@tidy arrayOf(output.tfjsArray)
    }.first()
    return rawOutput.toNDArray() as T
}
