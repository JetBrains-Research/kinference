package org.jetbrains.research.kotlin.mpp.inference.math.extensions

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*

fun splitByZero(buffer: FloatBuffer, strides: Strides, split: IntArray, keepDims: Boolean): Array<NDBuffer<Float>> {
    val array = buffer.array
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
        val newStrides = TensorStrides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun splitByZero(buffer: DoubleBuffer, strides: Strides, split: IntArray, keepDims: Boolean): Array<NDBuffer<Double>> {
    val array = buffer.array
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
        val newStrides = TensorStrides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun splitByZero(buffer: IntBuffer, strides: Strides, split: IntArray, keepDims: Boolean): Array<NDBuffer<Int>> {
    val array = buffer.array
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
        val newStrides = TensorStrides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun splitByZero(buffer: LongBuffer, strides: Strides, split: IntArray, keepDims: Boolean): Array<NDBuffer<Long>> {
    val array = buffer.array
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
        val newStrides = TensorStrides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun splitByZero(buffer: ShortBuffer, strides: Strides, split: IntArray, keepDims: Boolean): Array<NDBuffer<Short>> {
    val array = buffer.array
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
        val newStrides = TensorStrides(newShape)
        array.copyInto(newArray, 0, offset, offset + splitSize * split[it])
        offset += splitSize * split[it]
        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun <T : Any> NDBuffer<T>.splitByZero(split: IntArray, keepDims: Boolean): Array<NDBuffer<T>> {
    require(shape.size >= 2)
    return when (buffer) {
        is IntBuffer -> splitByZero(buffer as IntBuffer, strides, split, keepDims)
        is FloatBuffer -> splitByZero(buffer as FloatBuffer, strides, split, keepDims)
        is ShortBuffer -> splitByZero(buffer as ShortBuffer, strides, split, keepDims)
        is DoubleBuffer -> splitByZero(buffer as DoubleBuffer, strides, split, keepDims)
        is LongBuffer -> splitByZero(buffer as LongBuffer, strides, split, keepDims)
        else -> throw UnsupportedOperationException()
    } as Array<NDBuffer<T>>
}
