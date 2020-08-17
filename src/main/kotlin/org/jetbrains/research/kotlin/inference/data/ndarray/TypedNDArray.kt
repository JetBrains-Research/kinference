package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.broadcastShape
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createMutableNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createNDArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

interface TypedNDArray<T> {
    val type: TensorProto.DataType
    val array: T
    val strides: Strides
    val offset: Int

    val rank: Int
    val linearSize: Int
    val shape: IntArray

    operator fun get(i: Int): Any
    operator fun get(indices: IntArray): Any

    operator fun plus(other: TypedNDArray<T>) = plus(other, allocateHelper(shape, other.shape, type))
    fun plus(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>): TypedNDArray<T>

    operator fun minus(other: TypedNDArray<T>) = minus(other, allocateHelper(shape, other.shape, type))
    fun minus(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>): TypedNDArray<T>

    operator fun times(other: TypedNDArray<T>) = times(other, allocateHelper(shape, other.shape, type))
    fun times(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>): TypedNDArray<T>

    operator fun div(other: TypedNDArray<T>) = div(other, allocateHelper(shape, other.shape, type))
    fun div(other: TypedNDArray<T>, destination: MutableTypedNDArray<T>): TypedNDArray<T>

    fun clone(): TypedNDArray<T>
    fun row(row: Int): TypedNDArray<T>
    fun slice(starts: IntArray, ends: IntArray, steps: IntArray): TypedNDArray<T>
    fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int)
    fun toMutable(): MutableTypedNDArray<T>

    fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<T>
    fun slice(sliceLength: Int, start: Int): T
    infix fun matmul(other: TypedNDArray<T>): TypedNDArray<T>

    fun view(vararg axes: Int): TypedNDArray<T> {
        val (additionalOffset, newShape) = viewHelper(axes, strides)
        return createNDArray(type, array, Strides(newShape), offset + additionalOffset)
    }
}

interface MutableTypedNDArray<T> : TypedNDArray<T> {
    operator fun set(i: Int, value: Any)

    operator fun plusAssign(other: TypedNDArray<T>)
    operator fun minusAssign(other: TypedNDArray<T>)
    operator fun timesAssign(other: TypedNDArray<T>)
    operator fun divAssign(other: TypedNDArray<T>)

    fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int)
    fun placeAll(startOffset: Int, block: Any?)

    fun reshape(shape: IntArray): MutableTypedNDArray<T> = reshape(Strides(shape))
    fun reshape(strides: Strides): MutableTypedNDArray<T>

    fun viewMutable(vararg axes: Int): MutableTypedNDArray<T> {
        val (additionalOffset, newShape) = viewHelper(axes, strides)
        return createMutableNDArray(type, array, Strides(newShape), offset + additionalOffset)
    }

    fun clean()
}

fun viewHelper(axes: IntArray, strides: Strides): Pair<Int, IntArray> {
    val newOffset = strides.strides.reduceIndexed { index, acc, i -> acc + i * axes[index] }
    val newShape = strides.shape.copyOfRange(axes.size, strides.shape.size)

    return newOffset to newShape
}

fun <T> allocateHelper(shape: IntArray, otherShape: IntArray, type: TensorProto.DataType): MutableTypedNDArray<T> {
    return if (shape.contentEquals(otherShape))
        allocateNDArray(type, Strides(shape))
    else
        allocateNDArray(type, Strides(broadcastShape(shape, otherShape)))
}
