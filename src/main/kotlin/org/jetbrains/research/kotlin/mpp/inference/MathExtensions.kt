package org.jetbrains.research.kotlin.mpp.inference

import scientifik.kmath.structures.MutableBuffer

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
