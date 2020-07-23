package org.jetbrains.research.kotlin.inference.extensions

import scientifik.kmath.structures.*

fun minus(left: FloatBuffer, right: FloatBuffer): FloatBuffer {
    return plus(left, -right)
}


fun minus(left: IntBuffer, right: IntBuffer): IntBuffer {
    return plus(left, -right)
}

fun minus(left: LongBuffer, right: LongBuffer): LongBuffer {
    return plus(left, -right)
}

fun minus(left: DoubleBuffer, right: DoubleBuffer): DoubleBuffer {
    return plus(left, -right)
}

fun minus(left: ShortBuffer, right: ShortBuffer): ShortBuffer {
    return plus(left, -right)
}

@Suppress("UNCHECKED_CAST")
operator fun <T> NDBuffer<T>.minus(other: NDBuffer<T>): NDBuffer<T> {
    require(this::class == other::class)
    require(this.shape.contentEquals(other.shape))
    return when (buffer) {
        is IntBuffer -> BufferNDStructure(strides, minus(this.buffer as IntBuffer, other.buffer as IntBuffer))
        is FloatBuffer -> BufferNDStructure(strides, minus(this.buffer as FloatBuffer, other.buffer as FloatBuffer))
        is ShortBuffer -> BufferNDStructure(strides, minus((buffer as ShortBuffer), other.buffer as ShortBuffer))
        is DoubleBuffer -> BufferNDStructure(strides, minus(this.buffer as DoubleBuffer, other.buffer as DoubleBuffer))
        is LongBuffer -> BufferNDStructure(strides, minus(this.buffer as LongBuffer, other.buffer as LongBuffer))
        else -> throw UnsupportedOperationException()
    } as NDBuffer<T>
}

@Suppress("UNCHECKED_CAST")
fun <T : Any> NDBuffer<T>.minusScalar(x: T): NDBuffer<T> {
    return when (buffer) {
        is IntBuffer -> BufferNDStructure(strides, minus(this.buffer as IntBuffer, IntArray(this.buffer.size) { x as Int }.asBuffer()))
        is FloatBuffer -> BufferNDStructure(strides, minus(this.buffer as FloatBuffer, FloatArray(this.buffer.size) { x as Float }.asBuffer()))
        is ShortBuffer -> BufferNDStructure(strides, minus((buffer as ShortBuffer), ShortArray(this.buffer.size) { x as Short }.asBuffer()))
        is DoubleBuffer -> BufferNDStructure(strides, minus(this.buffer as DoubleBuffer, DoubleArray(this.buffer.size) { x as Double }.asBuffer()))
        is LongBuffer -> BufferNDStructure(strides, minus(this.buffer as LongBuffer, LongArray(this.buffer.size) { x as Long }.asBuffer()))
        else -> throw UnsupportedOperationException()
    } as NDBuffer<T>
}
