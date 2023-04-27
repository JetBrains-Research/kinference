package io.kinference.ndarray.extensions.abs

import io.kinference.primitives.types.PrimitiveType
import kotlin.math.abs

internal fun abs(x: Short) = abs(x.toInt()).toShort()
internal fun abs(x: Byte) = abs(x.toInt()).toByte()

internal inline fun abs(x: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
