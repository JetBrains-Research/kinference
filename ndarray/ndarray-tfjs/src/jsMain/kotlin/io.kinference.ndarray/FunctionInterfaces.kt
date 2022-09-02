package io.kinference.ndarray

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.moments
import io.kinference.ndarray.extensions.toNDArray
import io.ktor.utils.io.core.*

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
