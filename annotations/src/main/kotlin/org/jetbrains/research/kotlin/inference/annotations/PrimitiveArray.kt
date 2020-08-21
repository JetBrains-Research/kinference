@file:Suppress("UNUSED_PARAMETER")

package org.jetbrains.research.kotlin.inference.annotations

enum class DataType {
    BYTE,
    SHORT,
    INT,
    LONG,

    UBYTE,
    USHORT,
    UINT,
    ULONG,

    FLOAT,
    DOUBLE,

    UNKNOWN
}

class PrimitiveType {
    companion object {
        val MIN_VALUE: PrimitiveType = PrimitiveType()
        val MAX_VALUE: PrimitiveType = PrimitiveType()

        const val SIZE_BYTES: Int = 0
        const val SIZE_BITS: Int = 0
    }

    init {
        throw IllegalStateException("Don't use this class in runtime")
    }

    operator fun plus(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
    operator fun minus(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
    operator fun times(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
    operator fun div(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
    operator fun rem(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

    operator fun inc(): PrimitiveType = throw UnsupportedOperationException()
    operator fun dec(): PrimitiveType = throw UnsupportedOperationException()
    operator fun unaryPlus(): PrimitiveType = throw UnsupportedOperationException()
    operator fun unaryMinus(): PrimitiveType = throw UnsupportedOperationException()

    operator fun compareTo(other: PrimitiveType): Int = throw UnsupportedOperationException()

    fun toPrimitive(): PrimitiveType = throw UnsupportedOperationException()
}

fun Int.toPrimitive(): PrimitiveType = throw UnsupportedOperationException()

class PrimitiveArray(val size: Int) {
    constructor(size: Int, init: (Int) -> Any) : this(size)

    init {
        throw IllegalStateException("Don't use this class in runtime")
    }

    val indices: IntRange = 0 until size

    operator fun get(index: Int): PrimitiveType = throw UnsupportedOperationException()
    operator fun set(index: Int, value: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

    fun min(): PrimitiveType = throw UnsupportedOperationException()
    fun max(): PrimitiveType = throw UnsupportedOperationException()
    fun sum(): PrimitiveType = throw UnsupportedOperationException()

    fun fill(element: PrimitiveType, fromIndex: Int = 0, toIndex: Int = size): Unit = throw UnsupportedOperationException()

    fun sliceArray(indices: IntRange): PrimitiveArray = throw UnsupportedOperationException()

    fun copyOf(): PrimitiveArray = throw UnsupportedOperationException()
    fun copyOfRange(fromIndex: Int, toIndex: Int): PrimitiveArray = throw UnsupportedOperationException()
    fun copyInto(destination: PrimitiveArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size): PrimitiveArray = throw UnsupportedOperationException()
}
