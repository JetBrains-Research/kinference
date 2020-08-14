package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

class BooleanNDArray(array: BooleanArray, strides: Strides = Strides.empty(), offset: Int = 0) : NDArray<BooleanArray>(array, strides, TensorProto.DataType.BOOL, offset) {
    override fun clone(newStrides: Strides): BooleanNDArray {
        return BooleanNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Boolean {
        return array[i]
    }

    override fun get(indices: IntArray): Boolean {
        return array[strides.offset(indices)]
    }

    override fun set(i: Int, value: Any) {
        array[i] = value as Boolean
    }

    override fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int) {
        array as LateInitBooleanArray
        for (index in range) {
            array.putNext(this.array[offset + index])
        }
    }

    override fun plus(other: NDArray<BooleanArray>, destination: NDArray<BooleanArray>?): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun minus(other: NDArray<BooleanArray>, destination: NDArray<BooleanArray>?): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun times(other: NDArray<BooleanArray>, destination: NDArray<BooleanArray>?): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun div(other: NDArray<BooleanArray>, destination: NDArray<BooleanArray>?): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun mapElements(func: PrimitiveArrayFunction, copy: Boolean): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun clean() {
        TODO("Not yet implemented")
    }

    override fun slice(sliceLength: Int, start: Int): BooleanArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int) {
        block as BooleanArray
        block.copyInto(array, startOffset, startIndex, endIndex)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as BooleanArray
        block.copyInto(array, startOffset)
    }
}
