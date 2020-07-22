package org.jetbrains.research.kotlin.mpp.inference.mathExtension

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*

fun split(buffer: FloatBuffer, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean) : Array<NDBuffer<Float>> {
    val array = buffer.array

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
        val newStrides = TensorStrides(newShape)

        val newArray = FloatArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun split(buffer: DoubleBuffer, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean) : Array<NDBuffer<Double>> {
    val array = buffer.array

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
        val newStrides = TensorStrides(newShape)

        val newArray = DoubleArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun split(buffer: IntBuffer, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean) : Array<NDBuffer<Int>> {
    val array = buffer.array

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
        val newStrides = TensorStrides(newShape)

        val newArray = IntArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun split(buffer: LongBuffer, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean) : Array<NDBuffer<Long>> {
    val array = buffer.array

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
        val newStrides = TensorStrides(newShape)

        val newArray = LongArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun split(buffer: ShortBuffer, axis: Int, strides: Strides, split: IntArray, keepDims: Boolean) : Array<NDBuffer<Short>> {
    val array = buffer.array

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
        val newStrides = TensorStrides(newShape)

        val newArray = ShortArray(newStrides.linearSize)
        val factor = num * (split.getOrNull(num - 1) ?: 0)

        for (i in newArray.indices) {
            val indices = newStrides.index(i)
            indices[axis] += factor
            newArray[i] = array[strides.offset(indices)]
        }

        BufferNDStructure(newStrides, newArray.asBuffer())
    }
}

fun <T : Any> NDBuffer<T>.split(axis: Int, split: IntArray, keepDims: Boolean): Array<NDBuffer<T>> {
    return when(buffer) {
        is IntBuffer -> split(buffer as IntBuffer, axis, strides, split, keepDims)
        is FloatBuffer -> split(buffer as FloatBuffer, axis, strides, split, keepDims)
        is ShortBuffer -> split(buffer as ShortBuffer, axis, strides, split, keepDims)
        is DoubleBuffer -> split(buffer as DoubleBuffer, axis, strides, split, keepDims)
        is LongBuffer -> split(buffer as LongBuffer, axis, strides, split, keepDims)
        else -> throw UnsupportedOperationException()
    } as Array<NDBuffer<T>>
}
