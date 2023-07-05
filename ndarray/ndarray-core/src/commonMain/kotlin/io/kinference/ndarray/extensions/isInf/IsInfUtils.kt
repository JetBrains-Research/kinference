package io.kinference.ndarray.extensions.isInf

import io.kinference.primitives.types.PrimitiveType

internal val PrimitiveType.Companion.POSITIVE_INFINITY: PrimitiveType
    get() = throw UnsupportedOperationException()

internal val PrimitiveType.Companion.NEGATIVE_INFINITY: PrimitiveType
    get() = throw UnsupportedOperationException()

fun PrimitiveType.isInfinite(): Boolean = throw UnsupportedOperationException()
