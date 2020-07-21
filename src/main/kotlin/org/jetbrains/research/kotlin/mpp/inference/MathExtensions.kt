package org.jetbrains.research.kotlin.mpp.inference

import scientifik.kmath.linear.BufferMatrix
import scientifik.kmath.linear.dot
import scientifik.kmath.structures.Buffer
import scientifik.kmath.structures.DoubleBuffer
import scientifik.kmath.structures.MutableBuffer
import scientifik.kmath.structures.array

fun FloatArray.asBuffer() = FloatBuffer(this)

inline class FloatBuffer(val array: FloatArray) : MutableBuffer<Float> {
    override val size: Int get() = array.size

    override fun get(index: Int): Float = array[index]

    override fun set(index: Int, value: Float) {
        array[index] = value
    }

    override fun iterator() = array.iterator()

    override fun copy(): MutableBuffer<Float> = FloatBuffer(array.copyOf())
}


val Buffer<out Float>.array: FloatArray
    get() = if (this is FloatBuffer) {
        array
    } else {
        FloatArray(size) { get(it) }
    }

infix fun BufferMatrix<Float>.dot(other: BufferMatrix<Float>): BufferMatrix<Float> {
    if (this.colNum != other.rowNum) error("Matrix dot operation dimension mismatch: ($rowNum, $colNum) x (${other.rowNum}, ${other.colNum})")

    val array = FloatArray(this.rowNum * other.colNum)

    val a = this.buffer.array
    val b = other.buffer.array


    for (i in (0 until rowNum)) {
        for (j in (0 until other.colNum)) {
            for (k in (0 until colNum)) {
                array[i * other.colNum + j] += a[i * colNum + k] * b[k * other.colNum + j]
            }
        }
    }

    val buffer = FloatBuffer(array)
    return BufferMatrix(rowNum, other.colNum, buffer)
}

@Suppress("FunctionName")
inline fun FloatBuffer(size: Int, init: (Int) -> Float) = FloatBuffer(FloatArray(size) { init(it) })
