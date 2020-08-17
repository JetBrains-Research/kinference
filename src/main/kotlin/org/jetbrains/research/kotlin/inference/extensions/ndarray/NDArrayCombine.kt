package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.data.tensors.broadcast
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayValueCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayWithScalar
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> TypedNDArray<T>.combine(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>, ordered: Boolean = true) {
    if (other.isScalar()) {
        return this.scalarCombine(other[0] as V, destination, func)
    } else if (!ordered && this.isScalar()) {
        return other.scalarCombine(this[0] as V, destination, func)
    }

    func as PrimitiveArraysCombineFunction<T>
    if (!shape.contentEquals(other.shape)) {
        return this.applyWithBroadcast(other, destination, func)
    }

    func.apply(array, offset, other.array, other.offset, destination.array, destination.offset, linearSize)
    //return NDArray(type, func.apply(array, other.array), strides, false)
}

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> MutableTypedNDArray<T>.combineAssign(other: TypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    if (other.isScalar()) {
        return this.scalarCombineAssign(other[0] as V, func)
    }

    func as PrimitiveArraysCombineFunction<T>
    val actualOther = if (!shape.contentEquals(other.shape)) other.broadcast(this.shape) else other
    func.apply(array, offset, actualOther.array, actualOther.offset, array, offset, linearSize)

    return this
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarCombine(x: V, destination: MutableTypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>) {
    func as PrimitiveArrayWithScalar<T, V>
    func.apply(array, offset, x, destination.array, destination.offset, linearSize)
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarCombineAssign(x: V, op: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    op as PrimitiveArrayWithScalar<T, V>
    op.apply(array, offset, x, array, offset, linearSize)
    return this
}
