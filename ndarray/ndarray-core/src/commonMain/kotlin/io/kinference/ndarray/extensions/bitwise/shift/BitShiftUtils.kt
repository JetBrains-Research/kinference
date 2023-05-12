package io.kinference.ndarray.extensions.bitwise.shift

import io.kinference.primitives.types.PrimitiveType

internal fun PrimitiveType.shl(bitCount: Int): PrimitiveType = throw UnsupportedOperationException()

internal fun PrimitiveType.shr(bitCount: Int): PrimitiveType = throw UnsupportedOperationException()

internal fun UShort.shl(bitCount: Int) = this.toUInt().shl(bitCount).toUShort()

internal fun UShort.shr(bitCount: Int) = this.toUInt().shr(bitCount).toUShort()

internal fun UByte.shl(bitCount: Int) = this.toUInt().shl(bitCount).toUByte()

internal fun UByte.shr(bitCount: Int) = this.toUInt().shr(bitCount).toUByte()
