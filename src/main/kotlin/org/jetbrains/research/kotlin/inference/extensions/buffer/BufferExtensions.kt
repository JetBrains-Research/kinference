@file:Suppress("IMPLICIT_CAST_TO_ANY", "UNCHECKED_CAST")

package org.jetbrains.research.kotlin.inference.extensions.buffer

import TensorProto
import scientifik.kmath.structures.*
import kotlin.math.exp

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

@Suppress("FunctionName")
inline fun FloatBuffer(size: Int, init: (Int) -> Float) = FloatBuffer(FloatArray(size) { init(it) })

val Buffer<out Float>.array: FloatArray
    get() = if (this is FloatBuffer) {
        array
    } else {
        FloatArray(size) { get(it) }
    }

val Buffer<out Long>.array: LongArray
    get() = if (this is LongBuffer) {
        array
    } else {
        LongArray(size) { get(it) }
    }

val Buffer<out Short>.array: ShortArray
    get() = if (this is ShortBuffer) {
        array
    } else {
        ShortArray(size) { get(it) }
    }


val SUPPORTED_TYPES = setOf(TensorProto.DataType.DOUBLE, TensorProto.DataType.FLOAT, TensorProto.DataType.INT64, TensorProto.DataType.INT32, TensorProto.DataType.INT16)
fun inferType(type1: TensorProto.DataType, type2: TensorProto.DataType): TensorProto.DataType {
    return when {
        type1 !in SUPPORTED_TYPES || type2 !in SUPPORTED_TYPES -> error("Unsupported type")
        type1 == TensorProto.DataType.DOUBLE || type2 == TensorProto.DataType.DOUBLE -> TensorProto.DataType.DOUBLE
        type1 == TensorProto.DataType.FLOAT || type2 == TensorProto.DataType.FLOAT -> TensorProto.DataType.FLOAT
        type1 == TensorProto.DataType.INT64 || type2 == TensorProto.DataType.INT64 -> TensorProto.DataType.INT64
        type1 == TensorProto.DataType.INT32 || type2 == TensorProto.DataType.INT32 -> TensorProto.DataType.INT32
        type1 == TensorProto.DataType.INT16 || type2 == TensorProto.DataType.INT16 -> TensorProto.DataType.INT16
        else -> error("Unsupported type")
    }
}

inline fun <reified T> createInferredTypeBuffer(type1: TensorProto.DataType, type2: TensorProto.DataType, size: Int, noinline init: (Int) -> T): Pair<Buffer<T>, TensorProto.DataType> {
    val inferred = inferType(type1, type2)
    return createBuffer(inferred, size, init) to inferred
}

inline fun <reified T> createBuffer(type: TensorProto.DataType, size: Int, noinline init: (Int) -> T): Buffer<T> {
    return when (type) {
        TensorProto.DataType.DOUBLE -> DoubleBuffer(DoubleArray(size) { (init(it) as Number).toDouble() })
        TensorProto.DataType.FLOAT -> FloatBuffer(FloatArray(size) { (init(it) as Number).toFloat() })
        TensorProto.DataType.INT64 -> LongBuffer(LongArray(size) { (init(it) as Number).toLong() })
        TensorProto.DataType.INT32 -> IntBuffer(IntArray(size) { (init(it) as Number).toInt() })
        TensorProto.DataType.INT16 -> ShortBuffer(ShortArray(size) { (init(it) as Number).toShort() })
        else -> ArrayBuffer(Array(size, init))
    } as Buffer<T>
}

inline fun allocateMutableBuffer(type: TensorProto.DataType, size: Int): MutableBuffer<Any?> {
    return when (type) {
        TensorProto.DataType.DOUBLE -> DoubleBuffer(DoubleArray(size))
        TensorProto.DataType.FLOAT -> FloatBuffer(FloatArray(size))
        TensorProto.DataType.INT64 -> LongBuffer(LongArray(size))
        TensorProto.DataType.INT32 -> IntBuffer(IntArray(size))
        TensorProto.DataType.INT16 -> ShortBuffer(ShortArray(size))
        else -> ArrayBuffer(Array<Any?>(size) { null }) //FIXME: workaround other cases
    } as MutableBuffer<Any?>
}

inline fun <reified T> MutableBuffer<T?>.placeAll(buffer: Buffer<Any>, index: Int = 0) {
    for (i in 0 until buffer.size)
        this[i + index] = buffer[i] as T
}

inline fun zerosBuffer(type: TensorProto.DataType, size: Int): MutableBuffer<Any?> {
    return when (type) {
        TensorProto.DataType.DOUBLE -> DoubleArray(size) { 0.0 }.asBuffer()
        TensorProto.DataType.FLOAT -> FloatArray(size) { 0.0f }.asBuffer()
        TensorProto.DataType.INT64 -> LongArray(size) { 0L }.asBuffer()
        TensorProto.DataType.INT32 -> IntArray(size) { 0 }.asBuffer()
        TensorProto.DataType.INT16 -> ShortArray(size) { (0).toShort() }.asBuffer()
        else -> ArrayBuffer(Array<Any?>(size) { null })
    } as MutableBuffer<Any?>
}

inline fun <reified T> NDBuffer<T>.max(): T? {
    return when (buffer) {
        is IntBuffer -> (this.buffer as IntBuffer).array.max()
        is FloatBuffer -> (this.buffer as FloatBuffer).array.max()
        is ShortBuffer -> (this.buffer as ShortBuffer).array.max()
        is DoubleBuffer -> (this.buffer as DoubleBuffer).array.max()
        is LongBuffer -> (this.buffer as LongBuffer).array.max()
        else -> throw UnsupportedOperationException()
    } as? T
}

inline fun <reified T> NDBuffer<T>.sum(): T {
    return when (buffer) {
        is IntBuffer -> (this.buffer as IntBuffer).array.sum()
        is FloatBuffer -> (this.buffer as FloatBuffer).array.sum()
        is ShortBuffer -> (this.buffer as ShortBuffer).array.sum()
        is DoubleBuffer -> (this.buffer as DoubleBuffer).array.sum()
        is LongBuffer -> (this.buffer as LongBuffer).array.sum()
        else -> throw UnsupportedOperationException()
    } as T
}

inline fun <reified T> NDBuffer<T>.exp(): NDBuffer<T> {
    return when (buffer) {
        is FloatBuffer -> BufferNDStructure(strides, (buffer as FloatBuffer).array.apply { for (i in this.indices) this[i] = exp(this[i]) }.asBuffer())
        is DoubleBuffer -> BufferNDStructure(strides, (buffer as DoubleBuffer).array.apply { for (i in this.indices) this[i] = exp(this[i]) }.asBuffer())
        else -> throw UnsupportedOperationException()
    } as NDBuffer<T>
}
