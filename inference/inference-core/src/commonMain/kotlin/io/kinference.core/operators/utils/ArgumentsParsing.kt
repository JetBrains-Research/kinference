package io.kinference.core.operators.utils

import io.kinference.ndarray.extensions.utils.divCeil

public fun parseStrides(strides: IntArray?, shapeSize: Int) = strides ?: IntArray(shapeSize - 2) { 1 }

public fun parseDilations(dilations: IntArray?, shapeSize: Int) = dilations ?: IntArray(shapeSize - 2) { 1 }

public fun parsePads(autoPad: String, pads: IntArray?, xShape: IntArray, wShape: IntArray, strides: IntArray): IntArray {
    if (autoPad != "NOTSET")
        require(pads == null) { "Explicit pads cannot be used simultaneously with auto_pad attribute." }

    val size = xShape.size - 2

    return when (autoPad) {
        "SAME_UPPER" -> {
            IntArray(size * 2) { i ->
                val outputShape = xShape[i % size + 2] divCeil strides[i % size]
                var pad = (outputShape - 1) * strides[i % size] + wShape[i % size + 2] - xShape[i % size + 2]
                if (i >= size && pad % 2 == 1)
                    pad++
                pad / 2
            }
        }

        "SAME_LOWER" -> {
            IntArray(size * 2) { i ->
                val outputShape = xShape[i % size + 2] divCeil strides[i % size]
                var pad = (outputShape - 1) * strides[i % size] + wShape[i % size + 2] - xShape[i % size + 2]
                if (i < size && pad % 2 == 1) // not necessary to check whether pad is odd or not
                    pad++
                pad / 2
            }
        }

        "VALID" -> {
            xShape.forEachIndexed { index, i -> require(i >= wShape[index]) }

            IntArray(xShape.size * 2)
        }

        "NOTSET" -> {
            if (pads == null)
                IntArray(size * 2)
            else
                pads as IntArray
        }

        else -> throw IllegalArgumentException("Invalid auto_pad argument: $autoPad.")
    }
}
