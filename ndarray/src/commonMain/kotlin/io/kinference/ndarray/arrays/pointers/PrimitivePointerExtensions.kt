package io.kinference.ndarray.arrays.pointers

import io.kinference.primitives.types.PrimitiveType

inline fun IntPointer.acceptWithRecursive(src: PrimitivePointer, rec: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType, rec: PrimitiveType) -> Int) {

}

inline fun IntPointer.accept(src: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType) -> Int) {

}

inline fun PrimitivePointer.isCompatibleWith(other: IntPointer): Boolean = throw UnsupportedOperationException()
