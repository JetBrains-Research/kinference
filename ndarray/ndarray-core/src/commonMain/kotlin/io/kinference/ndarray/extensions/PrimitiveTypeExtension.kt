package io.kinference.ndarray.extensions

import io.kinference.primitives.types.PrimitiveType

internal inline fun min(a: PrimitiveType, b: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
internal inline fun max(a: PrimitiveType, b: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
internal inline fun exp(x: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

internal val PrimitiveType.absoluteValue: PrimitiveType
    get() = throw UnsupportedOperationException()

internal fun PrimitiveType.pow(n: Int): PrimitiveType = throw UnsupportedOperationException()
internal fun PrimitiveType.pow(n: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
internal fun PrimitiveType.withSign(sign: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

internal val PrimitiveType.sign: PrimitiveType
    get() = throw UnsupportedOperationException()
