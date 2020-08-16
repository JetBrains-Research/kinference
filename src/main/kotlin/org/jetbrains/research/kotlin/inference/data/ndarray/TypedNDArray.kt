package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

interface TypedNDArray<T> {
    val type: TensorProto.DataType
    val array: T
    val strides: Strides

    val rank: Int
    val linearSize: Int
    val shape: IntArray

    operator fun get(i: Int): Any
    operator fun get(indices: IntArray): Any
    operator fun plus(other: TypedNDArray<T>): TypedNDArray<T>
    operator fun minus(other: TypedNDArray<T>): TypedNDArray<T>
    operator fun times(other: TypedNDArray<T>): TypedNDArray<T>
    operator fun div(other: TypedNDArray<T>): TypedNDArray<T>

    fun clone(): TypedNDArray<T>
    fun row(row: Int): TypedNDArray<T>
    fun slice(starts: IntArray, ends: IntArray, steps: IntArray): TypedNDArray<T>
    fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int)
    fun toMutable(): MutableTypedNDArray<T>

    fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<T>
    fun slice(sliceLength: Int, start: Int): Any
    infix fun matmul(other: TypedNDArray<T>): TypedNDArray<T>
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

    fun clean()
}
