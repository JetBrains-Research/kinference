package org.jetbrains.research.kotlin.inference.math.extensions

/*import org.jetbrains.research.kotlin.inference.data.ndarray.MutableTypedNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.TypedNDArray
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayValueCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayWithScalar
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArraysCombineFunction

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> TypedNDArray<T>.combine(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>, ordered: Boolean = false): MutableTypedNDArray<T> {
    if (other.isScalar()) {
        return this.scalarCombine(other[0] as V, destination, func)
    } else if (!ordered && this.isScalar()) {
        return other.scalarCombine(this[0] as V, destination, func)
    }

    func as PrimitiveArraysCombineFunction<T>
    if (!shape.contentEquals(other.shape)) {
        return this.applyWithBroadcast(other, destination, func, ordered)
    }

    func.apply(array, offset, other.array, other.offset, destination.array, destination.offset, linearSize)
    return destination
}

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> MutableTypedNDArray<T>.combineAssign(other: TypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    return this.combine(other, this, func, true)
}

fun <T : Any, V : Any> TypedNDArray<T>.scalarCombine(x: V, destination: MutableTypedNDArray<T>, func: PrimitiveArrayValueCombineFunction<T, V>): MutableTypedNDArray<T> {
    func as PrimitiveArrayWithScalar<T, V>
    func.apply(array, offset, x, destination.array, destination.offset, linearSize)

    return destination
}

fun <T : Any, V : Any> MutableTypedNDArray<T>.scalarCombineAssign(x: V, func: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    return this.scalarCombine(x, this, func)
}*/
