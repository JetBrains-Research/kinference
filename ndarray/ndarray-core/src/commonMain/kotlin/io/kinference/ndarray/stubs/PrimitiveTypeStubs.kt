@file:Suppress("UNUSED_PARAMETER", "UnusedReceiverParameter", "unused")

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

internal val PrimitiveType.Companion.MAX_VALUE_FOR_MIN: PrimitiveType
    get() = throw UnsupportedOperationException()
