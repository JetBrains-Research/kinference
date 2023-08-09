package io.kinference.ndarray.math

import io.kinference.primitives.types.PrimitiveType

expect object Math {
    fun floorMod(x: Int, y: Int): Int
    fun floorMod(x: Long, y: Long): Long
}

internal inline fun Math.floorMod(x: PrimitiveType, y: PrimitiveType): PrimitiveType = throw UnsupportedOperationException()
