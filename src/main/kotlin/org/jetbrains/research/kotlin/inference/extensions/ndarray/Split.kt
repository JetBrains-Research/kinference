package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun split(array: FloatArray, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean): Array<FloatNDArray> {
    return Array(split.size) { num ->
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[axis] = split[num]
        } else {
            newShape = IntArray(strides.shape.size - 1)
            strides.shape.copyInto(newShape, 0, 0, axis)
            strides.shape.copyInto(newShape, axis, axis + 1)
        }
        val newStrides = Strides(newShape)

        val newArray = FloatArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        FloatNDArray(newArray, newStrides)
    }
}

fun split(array: DoubleArray, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean): Array<DoubleNDArray> {
    return Array(split.size) { num ->
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[axis] = split[num]
        } else {
            newShape = IntArray(strides.shape.size - 1)
            strides.shape.copyInto(newShape, 0, 0, axis)
            strides.shape.copyInto(newShape, axis, axis + 1)
        }
        val newStrides = Strides(newShape)

        val newArray = DoubleArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }
        DoubleNDArray(newArray, newStrides)
    }
}

fun split(array: IntArray, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean): Array<IntNDArray> {
    return Array(split.size) { num ->
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[axis] = split[num]
        } else {
            newShape = IntArray(strides.shape.size - 1)
            strides.shape.copyInto(newShape, 0, 0, axis)
            strides.shape.copyInto(newShape, axis, axis + 1)
        }
        val newStrides = Strides(newShape)

        val newArray = IntArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        IntNDArray(newArray, newStrides)
    }
}

fun split(array: LongArray, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean): Array<LongNDArray> {
    return Array(split.size) { num ->
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[axis] = split[num]
        } else {
            newShape = IntArray(strides.shape.size - 1)
            strides.shape.copyInto(newShape, 0, 0, axis)
            strides.shape.copyInto(newShape, axis, axis + 1)
        }
        val newStrides = Strides(newShape)

        val newArray = LongArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        LongNDArray(newArray, newStrides)
    }
}

fun split(array: ShortArray, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean): Array<ShortNDArray> {
    return Array(split.size) { num ->
        val newShape: IntArray
        if (keepDims) {
            newShape = strides.shape.copyOf()
            newShape[axis] = split[num]
        } else {
            newShape = IntArray(strides.shape.size - 1)
            strides.shape.copyInto(newShape, 0, 0, axis)
            strides.shape.copyInto(newShape, axis, axis + 1)
        }
        val newStrides = Strides(newShape)

        val newArray = ShortArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        ShortNDArray(newArray, newStrides)
    }
}

fun NDArray.split(axis: Int, split: IntArray, keepDims: Boolean): Array<NDArray> {
    return when (array) {
        is IntArray -> split(array, axis, strides, split, keepDims)
        is FloatArray -> split(array, axis, strides, split, keepDims)
        is ShortArray -> split(array, axis, strides, split, keepDims)
        is DoubleArray -> split(array, axis, strides, split, keepDims)
        is LongArray -> split(array, axis, strides, split, keepDims)
        else -> throw UnsupportedOperationException()
    } as Array<NDArray>
}

fun NDArray.splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): Array<NDArray> {
    if (axis == 0 && rank >= 2) return splitByZero(split, keepDims)
    return split(axis, split, keepDims)
}
