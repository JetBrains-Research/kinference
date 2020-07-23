package org.jetbrains.research.kotlin.inference.extensions.math

import org.jetbrains.research.kotlin.inference.extensions.buffer.FloatBuffer
import org.jetbrains.research.kotlin.inference.extensions.buffer.asBuffer
import scientifik.kmath.structures.*

operator fun FloatBuffer.unaryMinus(): FloatBuffer {
    val array = FloatArray(this.size)

    val a = this.array

    for (i in (0 until this.size)) array[i] = -a[i]

    return array.asBuffer()
}


operator fun IntBuffer.unaryMinus(): IntBuffer {
    val array = IntArray(this.size)

    val a = this.array

    for (i in (0 until this.size)) array[i] = -a[i]

    return array.asBuffer()
}

operator fun LongBuffer.unaryMinus(): LongBuffer {
    val array = LongArray(this.size)

    val a = this.array

    for (i in (0 until this.size)) array[i] = -a[i]

    return array.asBuffer()
}

operator fun DoubleBuffer.unaryMinus(): DoubleBuffer {
    val array = DoubleArray(this.size)

    val a = this.array

    for (i in (0 until this.size)) array[i] = -a[i]

    return array.asBuffer()
}

operator fun ShortBuffer.unaryMinus(): ShortBuffer {
    val array = ShortArray(this.size)

    val a = this.array

    for (i in (0 until this.size)) array[i] = (-a[i]).toShort()

    return array.asBuffer()
}

operator fun <T : Any> NDBuffer<T>.unaryMinus(): NDBuffer<T> {
    return when (buffer) {
        is IntBuffer -> BufferNDStructure(strides, -(this.buffer as IntBuffer))
        is FloatBuffer -> BufferNDStructure(strides, -(this.buffer as FloatBuffer))
        is ShortBuffer -> BufferNDStructure(strides, -(this.buffer as ShortBuffer))
        is DoubleBuffer -> BufferNDStructure(strides, -(this.buffer as DoubleBuffer))
        is LongBuffer -> BufferNDStructure(strides, -(this.buffer as LongBuffer))
        else -> throw error("Unsupported operation")
    } as NDBuffer<T>
}
