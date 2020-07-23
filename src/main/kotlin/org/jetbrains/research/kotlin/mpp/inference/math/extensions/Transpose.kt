package org.jetbrains.research.kotlin.mpp.inference.math.extensions

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import scientifik.kmath.structures.*

fun transpose(buffer: FloatBuffer, strides: Strides, permutation: IntArray): NDBuffer<Float> {
    val oldArray = buffer.array
    val array = FloatArray(buffer.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = TensorStrides(newShape)


    for (i in array.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        array[i] = oldArray[strides.offset(newIndices)]
    }


    return BufferNDStructure(newStrides, array.asBuffer())
}

fun transpose(buffer: DoubleBuffer, strides: Strides, permutation: IntArray): NDBuffer<Double> {
    val oldArray = buffer.array
    val array = DoubleArray(buffer.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = TensorStrides(newShape)


    for (i in array.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        array[i] = oldArray[strides.offset(newIndices)]
    }


    return BufferNDStructure(newStrides, array.asBuffer())
}

fun transpose(buffer: IntBuffer, strides: Strides, permutation: IntArray): NDBuffer<Int> {
    val oldArray = buffer.array
    val array = IntArray(buffer.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = TensorStrides(newShape)


    for (i in array.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        array[i] = oldArray[strides.offset(newIndices)]
    }


    return BufferNDStructure(newStrides, array.asBuffer())
}

fun transpose(buffer: LongBuffer, strides: Strides, permutation: IntArray): NDBuffer<Long> {
    val oldArray = buffer.array
    val array = LongArray(buffer.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = TensorStrides(newShape)


    for (i in array.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        array[i] = oldArray[strides.offset(newIndices)]
    }


    return BufferNDStructure(newStrides, array.asBuffer())
}

fun transpose(buffer: ShortBuffer, strides: Strides, permutation: IntArray): NDBuffer<Short> {
    val oldArray = buffer.array
    val array = ShortArray(buffer.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = TensorStrides(newShape)


    for (i in array.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        array[i] = oldArray[strides.offset(newIndices)]
    }


    return BufferNDStructure(newStrides, array.asBuffer())
}

fun <T : Any> NDBuffer<T>.transpose(permutation: IntArray): NDBuffer<T> {
    return when (buffer) {
        is IntBuffer -> transpose(buffer as IntBuffer, strides, permutation)
        is FloatBuffer -> transpose(buffer as FloatBuffer, strides, permutation)
        is ShortBuffer -> transpose(buffer as ShortBuffer, strides, permutation)
        is DoubleBuffer -> transpose(buffer as DoubleBuffer, strides, permutation)
        is LongBuffer -> transpose(buffer as LongBuffer, strides, permutation)
        else -> throw UnsupportedOperationException()
    } as BufferNDStructure<T>
}
