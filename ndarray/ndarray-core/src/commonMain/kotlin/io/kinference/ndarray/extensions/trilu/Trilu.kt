package io.kinference.ndarray.extensions.trilu

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import kotlin.math.min
import kotlin.math.max

fun NDArray.trilu(k: Int, upper: Boolean): MutableNDArrayCore {
    val output = allocateNDArray(this.type, this.strides)
    val (height, width) = this.shape.takeLast(2)
    val batchSize = height * width
    val batchCount = this.computeBlockSize(toDim = this.rank - 2)
    if (upper) {
        for (batch in 0 until batchCount) {
            this.upperTrilu(batch * batchSize, k, height, width, output)
        }
    } else {
        for (batch in 0 until batchCount) {
            this.lowerTrilu(batch * batchSize, k, height, width, output)
        }
    }

    return output
}

private fun NDArray.upperTrilu(startOffset: Int, k: Int, height: Int, width: Int, dest: MutableNDArrayCore): NDArrayCore {
    if (k > width) return dest
    if (-k > width) {
        dest.copyFrom(offset = 0, this)
        return dest
    }

    val countNonZeroRows = min(height, width - k)
    for (rowIdx in 0 until countNonZeroRows) {
        val startCopyRange = startOffset + max(0, rowIdx * width + rowIdx + k)
        val endCopyRange = startOffset + (rowIdx + 1) * width
        dest.copyFrom(startCopyRange, this, startCopyRange, endCopyRange)
    }
    return dest
}

private fun NDArray.lowerTrilu(startOffset: Int, k: Int, height: Int, width: Int, dest: MutableNDArrayCore): NDArrayCore {
    if (-k > width) return dest
    if (k > width) {
        dest.copyFrom(offset = 0, this)
        return dest
    }

    val countNonZeroRows = min(height, width + k)
    for (rowIdx in height - countNonZeroRows  until height) {
        val startCopyRange = startOffset + width * rowIdx
        val endCopyRange = startCopyRange + min(width, rowIdx + k + 1)
        dest.copyFrom(startCopyRange, this, startCopyRange, endCopyRange)
    }
    return dest
}
