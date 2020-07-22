package org.jetbrains.research.kotlin.mpp.inference.mathExtension

import scientifik.kmath.structures.*

fun plus(left: FloatBuffer, right: FloatBuffer): FloatBuffer {
    require(left.size == right.size)

    val array = FloatArray(left.size)

    val a = left.array
    val b = right.array

    for (i in (0 until left.size)) array[i] = (a[i] + b[i])

    return array.asBuffer()
}


fun plus(left: IntBuffer, right: IntBuffer): IntBuffer {
    require(left.size == right.size)

    val array = IntArray(left.size)

    val a = left.array
    val b = right.array

    for (i in (0 until left.size)) array[i] = (a[i] + b[i])

    return array.asBuffer()
}

fun plus(left: LongBuffer, right: LongBuffer): LongBuffer {
    require(left.size == right.size)

    val array = LongArray(left.size)

    val a = left.array
    val b = right.array

    for (i in (0 until left.size)) array[i] = (a[i] + b[i])

    return array.asBuffer()
}

fun plus(left: DoubleBuffer, right: DoubleBuffer): DoubleBuffer {
    require(left.size == right.size)

    val array = DoubleArray(left.size)

    val a = left.array
    val b = right.array

    for (i in (0 until left.size)) array[i] = (a[i] + b[i])

    return array.asBuffer()
}

fun plus(left: ShortBuffer, right: ShortBuffer): ShortBuffer {
    require(left.size == right.size)

    val array = ShortArray(left.size)

    val a = left.array
    val b = right.array

    for (i in (0 until left.size)) array[i] = (a[i] + b[i]).toShort()

    return array.asBuffer()
}

fun <T : Any> NDBuffer<T>.plus(other: NDBuffer<T>): NDBuffer<T> {
    require(this::class == other::class)
    require(this.shape.contentEquals(other.shape))
    return when(buffer) {
        is IntBuffer -> BufferNDStructure(strides, plus(this.buffer as IntBuffer, other.buffer as IntBuffer))
        is FloatBuffer -> BufferNDStructure(strides, plus(this.buffer as FloatBuffer, other.buffer as FloatBuffer))
        is ShortBuffer -> BufferNDStructure(strides, plus((buffer as ShortBuffer), other.buffer as ShortBuffer))
        is DoubleBuffer -> BufferNDStructure(strides, plus(this.buffer as DoubleBuffer, other.buffer as DoubleBuffer))
        is LongBuffer -> BufferNDStructure(strides, plus(this.buffer as LongBuffer, other.buffer as LongBuffer))
        else -> throw UnsupportedOperationException()
    } as BufferNDStructure<T>
}
