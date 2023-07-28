package io.kinference.ndarray.stubs

import io.kinference.primitives.types.PrimitiveType

internal val PrimitiveType.absoluteValue: PrimitiveType
    get() = throw UnsupportedOperationException()

internal val PrimitiveType.sign: PrimitiveType
    get() = throw UnsupportedOperationException()

internal val PrimitiveType.Companion.POSITIVE_INFINITY: PrimitiveType
    get() = throw UnsupportedOperationException()

internal val PrimitiveType.Companion.NEGATIVE_INFINITY: PrimitiveType
    get() = throw UnsupportedOperationException()

internal fun PrimitiveType.isInfinite(): Boolean = throw UnsupportedOperationException()
internal fun PrimitiveType.isNaN(): Boolean = throw UnsupportedOperationException()

internal fun PrimitiveType.withSign(sign: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

internal fun PrimitiveType.pow(n: Int): PrimitiveType = throw UnsupportedOperationException()
internal fun PrimitiveType.pow(n: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

internal val PrimitiveType.Companion.MIN_VALUE_FOR_MAX: PrimitiveType
    get() = throw UnsupportedOperationException()

internal val Byte.Companion.MIN_VALUE_FOR_MAX: Byte
    get() = MIN_VALUE

internal val Short.Companion.MIN_VALUE_FOR_MAX: Short
    get() = MIN_VALUE

internal val Int.Companion.MIN_VALUE_FOR_MAX: Int
    get() = MIN_VALUE

internal val Long.Companion.MIN_VALUE_FOR_MAX: Long
    get() = MIN_VALUE

internal val UByte.Companion.MIN_VALUE_FOR_MAX: UByte
    get() = MIN_VALUE

internal val UShort.Companion.MIN_VALUE_FOR_MAX: UShort
    get() = MIN_VALUE

internal val UInt.Companion.MIN_VALUE_FOR_MAX: UInt
    get() = MIN_VALUE

internal val ULong.Companion.MIN_VALUE_FOR_MAX: ULong
    get() = MIN_VALUE

internal val Float.Companion.MIN_VALUE_FOR_MAX: Float
    get() = NEGATIVE_INFINITY

internal val Double.Companion.MIN_VALUE_FOR_MAX: Double
    get() = NEGATIVE_INFINITY
