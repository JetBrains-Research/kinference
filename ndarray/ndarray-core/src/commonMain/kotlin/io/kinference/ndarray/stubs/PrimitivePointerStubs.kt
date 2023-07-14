package io.kinference.ndarray.stubs

import io.kinference.ndarray.arrays.pointers.*
import io.kinference.primitives.types.PrimitiveType

internal inline fun IntPointer.acceptWithRecursive(src: PrimitivePointer, rec: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType, rec: PrimitiveType) -> Int) {
    throw UnsupportedOperationException()
}

internal inline fun FloatPointer.accept(src: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType) -> Float) {
    throw UnsupportedOperationException()
}

internal inline fun IntPointer.accept(src: PrimitivePointer, count: Int, action: (dst: Int, src: PrimitiveType) -> Int) {
    throw UnsupportedOperationException()
}

internal inline fun PrimitivePointer.isCompatibleWith(other: IntPointer): Boolean = throw UnsupportedOperationException()
internal inline fun PrimitivePointer.isCompatibleWith(other: LongPointer): Boolean = throw UnsupportedOperationException()

internal inline fun PrimitivePointer.forEachWith(other: LongPointer, count: Int, action: (value1: PrimitiveType, value2: Long) -> Unit) {
    throw UnsupportedOperationException()
}
