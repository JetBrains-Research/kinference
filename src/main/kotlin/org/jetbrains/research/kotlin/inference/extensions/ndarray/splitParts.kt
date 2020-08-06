package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.FloatNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun splitParts(array: FloatArray, parts: Int, strides: Strides): List<NDArray<FloatArray>> {
    require(array.size % parts == 0)
    require(strides.linearSize == array.size / parts)
    var offset = 0
    val partSize = strides.linearSize
    return List(parts) {
        val newArray = array.copyOfRange(offset, offset + partSize)
        offset += partSize
        FloatNDArray(newArray, strides)
    }
}
