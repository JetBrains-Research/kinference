package io.kinference.tfjs.utils

import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*

private suspend fun getIndices(indices: NumberNDArrayTFJS, axisLimit: Int): NDArrayTFJS {
    return tidyNDArray {
        val axisLimitScalar = NDArrayTFJS.intScalar(axisLimit)
        val zero = NDArrayTFJS.intScalar(0)
        val indicesGreaterOrEqualZero = indices.greaterEqual(zero)
        indices.where(indicesGreaterOrEqualZero, indices + axisLimitScalar)
    }
}

internal suspend fun NumberNDArrayTFJS.getFullIndices(axis: Int, axisLimit: Int, inputRank: Int): NumberNDArrayTFJS {
    return tidyNDArray {
        val actualIndices = getIndices(this, axisLimit)

        val reshapedIndices = actualIndices.reshape(intArrayOf(*actualIndices.shape, 1))
        val padArray = Array(reshapedIndices.rank) {
            if (it != reshapedIndices.rank - 1) {
                arrayOf(0, 0)
            } else {
                arrayOf(axis, inputRank - axis - 1)
            }
        }
        val paddedIndices = reshapedIndices.pad(padArray, 0) as NumberNDArrayTFJS

        // Add relevant values to indices for GatherND
        val baseRangeShape = Array(paddedIndices.rank) { 1 }
        val baseRangePad = Array(paddedIndices.rank) { arrayOf(0, 0) }

        val otherRelevantIndices = List(paddedIndices.rank - 1) { currentAxis ->
            if (currentAxis == axis) {
                // do nothing for operator axis
                null
            } else {
                // Make range for axis
                val range = NDArrayTFJS.intRange(0, paddedIndices.shape[currentAxis], 1)
                val rangeShape = baseRangeShape.copyOf().apply { set(currentAxis, paddedIndices.shape[currentAxis]) }
                //reshape to [1,...,paddedIndices.shape[axis],...,1]
                // reshapedRange.rank == paddedIndices.rank
                val reshapedRange = range.reshape(rangeShape.toIntArray())

                val rangePadding = baseRangePad.copyOf().apply { set(baseRangePad.lastIndex, arrayOf(currentAxis, inputRank - currentAxis - 1)) }
                // padding to [1,...,paddedIndices.shape[axis],...,input.rank]
                val paddedRange = reshapedRange.pad(rangePadding, 0)

                // broadcast to paddedIndices
                paddedRange.broadcastTo(paddedIndices.shapeArray)
            }
        }.filterNotNull().toTypedArray()

        // Adding relevant indices for GatherND
        paddedIndices.add(otherRelevantIndices)
    }
}
