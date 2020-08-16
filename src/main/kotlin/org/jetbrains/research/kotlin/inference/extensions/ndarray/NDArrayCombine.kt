package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.data.tensors.broadcast
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayValueCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> TypedNDArray<T>.combine(other: TypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>, ordered: Boolean = true): TypedNDArray<T> {
    if (other.isScalar()) {
        return this.scalarCombine(other[0] as V, func)
    } else if (!ordered && this.isScalar()) {
        return other.scalarCombine(this[0] as V, func)
    }

    func as PrimitiveArraysCombineFunction<T>
    if (!shape.contentEquals(other.shape)) {
        return this.applyWithBroadcast(other, func)
    }

    return NDArray(type, func.apply(array, other.array), strides, false)
}

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> TypedNDArray<T>.combineAssign(other: TypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    if (other.isScalar()) {
        return this.scalarCombineAssign(other[0] as V, func)
    }

    func as PrimitiveArraysCombineFunction<T>
    val actualOther = if (!shape.contentEquals(other.shape)) other.broadcast(this.shape) else other
    func.apply(array, actualOther.array)

    return this
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarCombine(x: V, func: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    return NDArray(type, func.apply(array, x), strides, false)
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarCombineAssign(x: V, op: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    op.apply(array, x)
    return this
}
