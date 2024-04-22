package io.kinference.utils.inlines

import io.kinference.primitives.types.PrimitiveType

infix fun PrimitiveType.or(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

infix fun PrimitiveType.and(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()

infix fun PrimitiveType.xor(other: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
