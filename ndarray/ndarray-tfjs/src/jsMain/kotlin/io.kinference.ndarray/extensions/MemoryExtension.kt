package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.core.tidy

fun tidy(fn: () -> Array<ArrayTFJS>): Array<ArrayTFJS> = tidy(fn, null)

fun tidyNDArrays(fn: () -> Array<NDArrayTFJS>): Array<NDArrayTFJS> {
    val rawOutput = tidy {
        val output = fn()
        return@tidy output.map { it.tfjsArray }.toTypedArray()
    }
    return rawOutput.map { it.toNDArray() }.toTypedArray()
}

fun tidyNDArray(fn: () -> NDArrayTFJS): NDArrayTFJS {
    val rawOutput = tidy {
        val output = fn()
        return@tidy arrayOf(output.tfjsArray)
    }.first()
    return rawOutput.toNDArray()
}
