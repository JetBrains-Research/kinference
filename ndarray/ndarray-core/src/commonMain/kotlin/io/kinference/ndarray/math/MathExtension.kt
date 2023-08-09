package io.kinference.ndarray.math

inline fun Math.floorMod(left: Byte, right: Byte) = floorMod(left.toInt(), right.toInt()).toByte()
inline fun Math.floorMod(left: Short, right: Short) = floorMod(left.toInt(), right.toInt()).toShort()
