package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

open class BooleanNDArray(array: BooleanArray, strides: Strides = Strides.empty()) : NDArray<BooleanArray>(array, strides, TensorProto.DataType.BOOL) {
    init {
        require(array.size == strides.linearSize)
    }

    override fun clone(): BooleanNDArray {
        return BooleanNDArray(array.copyOf(), strides)
    }

    override fun get(i: Int): Boolean {
        return array[i]
    }

    override fun get(indices: IntArray): Boolean {
        return array[strides.offset(indices)]
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitBooleanArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun div(other: TypedNDArray<BooleanArray>): TypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun mapElements(func: PrimitiveArrayFunction): TypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun slice(sliceLength: Int, start: Int): BooleanArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun plus(other: TypedNDArray<BooleanArray>): TypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun minus(other: TypedNDArray<BooleanArray>): TypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun times(other: TypedNDArray<BooleanArray>): TypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun toMutable(): MutableTypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }
}

class MutableBooleanNDArray(array: BooleanArray, strides: Strides = Strides.empty()) : BooleanNDArray(array, strides), MutableTypedNDArray<BooleanArray> {
    override fun set(i: Int, value: Any) {
        TODO("Not yet implemented")
    }

    override fun plusAssign(other: TypedNDArray<BooleanArray>) {
        TODO("Not yet implemented")
    }

    override fun minusAssign(other: TypedNDArray<BooleanArray>) {
        TODO("Not yet implemented")
    }

    override fun timesAssign(other: TypedNDArray<BooleanArray>) {
        TODO("Not yet implemented")
    }

    override fun divAssign(other: TypedNDArray<BooleanArray>) {
        TODO("Not yet implemented")
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        TODO("Not yet implemented")
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        TODO("Not yet implemented")
    }

    override fun reshape(strides: Strides): MutableTypedNDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun clean() {
        TODO("Not yet implemented")
    }
}
