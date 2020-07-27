package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun splitByZero(array: FloatArray, strides: Strides, split: IntArray, keepDims: Boolean): Array<FloatNDArray> {
    val splitSize = strides.strides[0]
    var offset = 0
    return Array(split.size) {
        val newArray = FloatArray(splitSize * split[it])
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[0] = split[it]
        } else {
            newShape = strides.shape.copyOfRange(1, strides.shape.size)
        }
        val newStrides = Strides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        FloatNDArray(newArray, newStrides)
    }
}

fun splitByZero(array: DoubleArray, strides: Strides, split: IntArray, keepDims: Boolean): Array<DoubleNDArray> {
    val splitSize = strides.strides[0]
    var offset = 0
    return Array(split.size) {
        val newArray = DoubleArray(splitSize * split[it])
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[0] = split[it]
        } else {
            newShape = strides.shape.copyOfRange(1, strides.shape.lastIndex)
        }
        val newStrides = Strides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        DoubleNDArray(newArray, newStrides)
    }
}

fun splitByZero(array: IntArray, strides: Strides, split: IntArray, keepDims: Boolean): Array<IntNDArray> {
    val splitSize = strides.strides[0]
    var offset = 0
    return Array(split.size) {
        val newArray = IntArray(splitSize * split[it])
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[0] = split[it]
        } else {
            newShape = strides.shape.copyOfRange(1, strides.shape.lastIndex)
        }
        val newStrides = Strides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        IntNDArray(newArray, newStrides)
    }
}

fun splitByZero(array: LongArray, strides: Strides, split: IntArray, keepDims: Boolean): Array<LongNDArray> {
    val splitSize = strides.strides[0]
    var offset = 0
    return Array(split.size) {
        val newArray = LongArray(splitSize * split[it])
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[0] = split[it]
        } else {
            newShape = strides.shape.copyOfRange(1, strides.shape.lastIndex)
        }
        val newStrides = Strides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        LongNDArray(newArray, newStrides)
    }
}

fun splitByZero(array: ShortArray, strides: Strides, split: IntArray, keepDims: Boolean): Array<ShortNDArray> {
    val splitSize = strides.strides[0]
    var offset = 0
    return Array(split.size) {
        val newArray = ShortArray(splitSize * split[it])
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[0] = split[it]
        } else {
            newShape = strides.shape.copyOfRange(1, strides.shape.lastIndex)
        }
        val newStrides = Strides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        ShortNDArray(newArray, newStrides)
    }
}

fun NDArray.splitByZero(split: IntArray, keepDims: Boolean): Array<NDArray> {
    require(shape.size >= 2)
    return when (array) {
        is IntArray -> splitByZero(array, strides, split, keepDims)
        is FloatArray -> splitByZero(array, strides, split, keepDims)
        is ShortArray -> splitByZero(array, strides, split, keepDims)
        is DoubleArray -> splitByZero(array, strides, split, keepDims)
        is LongArray -> splitByZero(array, strides, split, keepDims)
        else -> throw UnsupportedOperationException()
    } as Array<NDArray>
}
