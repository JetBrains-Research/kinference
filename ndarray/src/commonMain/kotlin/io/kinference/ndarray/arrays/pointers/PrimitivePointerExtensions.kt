package io.kinference.ndarray.arrays.pointers

import io.kinference.primitives.types.PrimitiveType

inline fun IntPointer.acceptWithRecursive(src: PrimitivePointer, rec: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType, rec: PrimitiveType) -> Int) = Unit

inline fun IntPointer.accept(src: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType) -> Int) = Unit

inline fun PrimitivePointer.isCompatibleWith(other: IntPointer): Boolean = throw UnsupportedOperationException()

inline fun PrimitivePointer.isCompatibleWith(other: LongPointer): Boolean = throw UnsupportedOperationException()

inline fun FloatPointer.accept(src: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType) -> Float) = Unit
