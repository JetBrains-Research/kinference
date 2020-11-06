package io.kinference.ndarray.arrays.pointers

import io.kinference.primitives.types.PrimitiveType

inline fun IntPointer.acceptWithRecursive(src: PrimitivePointer, rec: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType, rec: PrimitiveType) -> Int) {

}
