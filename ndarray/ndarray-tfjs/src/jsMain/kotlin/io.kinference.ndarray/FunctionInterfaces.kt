package io.kinference.ndarray

import io.kinference.ndarray.arrays.ArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.utils.Closeable

data class MomentsOutput(
    val mean: NumberNDArrayTFJS,
    val variance: NumberNDArrayTFJS
) : Closeable {
    override fun close() {
        mean.close()
        variance.close()
    }

    companion object {
        operator fun invoke(mean: ArrayTFJS, variance: ArrayTFJS): MomentsOutput {
            return MomentsOutput(NumberNDArrayTFJS(mean), NumberNDArrayTFJS(variance))
        }
    }
}
