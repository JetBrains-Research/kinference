package io.kinference.ndarray

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.utils.Closeable

external interface MomentsOutputTFJS {
    val mean: ArrayTFJS
    val variance: ArrayTFJS
}

data class MomentsOutput(
    val mean: NumberNDArrayTFJS,
    val variance: NumberNDArrayTFJS
) : Closeable {

    override fun close() {
        mean.close()
        variance.close()
    }
}

fun MomentsOutputTFJS.toNDArray() = MomentsOutput(NumberNDArrayTFJS(mean), NumberNDArrayTFJS(variance))
