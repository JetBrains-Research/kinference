package org.jetbrains.research.kotlin.mpp.inference.mathExtension

import TensorProto
import scientifik.kmath.linear.BufferMatrix
import scientifik.kmath.structures.*

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

val BufferMatrix<Float>.arrayRows: Array<FloatArray>
    get()= Array(rowNum) { i ->
        FloatArray(colNum) { j -> get(i, j) }
    }

@Suppress("FunctionName")
inline fun FloatBuffer(size: Int, init: (Int) -> Float) = FloatBuffer(FloatArray(size) { init(it) })

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

