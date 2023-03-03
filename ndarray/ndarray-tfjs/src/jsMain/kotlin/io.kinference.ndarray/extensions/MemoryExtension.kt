package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.core.*

suspend fun tidy(fn: suspend () -> Array<ArrayTFJS>): Array<ArrayTFJS> {
    val engine = engine()
    lateinit var result: Array<ArrayTFJS>
    return scopedRun(
        start = { engine.startScope(null) },
        end = { engine.endScope(result) },
        fn = {
            result = fn()
            result
        }
    )
}

@Suppress("UNCHECKED_CAST")
suspend fun <T : NDArrayTFJS> tidyNDArrays(fn: suspend () -> Array<T>): Array<T> {
    val rawOutput = tidy {
        val output = fn()
        return@tidy output.map { it.tfjsArray }.toTypedArray()
    }
    return rawOutput.map { it.toNDArray() as T }.toTypedArray()
}

@Suppress("UNCHECKED_CAST")
suspend fun <T : NDArrayTFJS> tidyNDArray(fn: suspend () -> T): T {
    val rawOutput = tidy {
        val output = fn()
        return@tidy arrayOf(output.tfjsArray)
    }.first()
    return rawOutput.toNDArray() as T
}

suspend fun scopedRun(start: () -> Unit, end: () -> Unit, fn: suspend () -> Array<ArrayTFJS>): Array<ArrayTFJS> {
    start()
    try {
        val res = fn()
        end();
        return res;
    } catch (e: Exception) {
        end()
        throw e;
    }
}
