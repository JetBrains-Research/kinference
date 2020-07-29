package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

class BooleanNDArray(array: BooleanArray, strides: Strides = Strides.empty()) : NDArray<BooleanArray>(array, strides, TensorProto.DataType.BOOL) {
    override fun clone(newStrides: Strides): BooleanNDArray {
        return BooleanNDArray(array.copyOf(), newStrides)
    }

    override fun get(i: Int): Boolean {
        return array[i]
    }

    override fun get(indices: IntArray): Boolean {
        return array[strides.offset(indices)]
    }

    override fun plus(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun times(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun div(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun minus(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as BooleanArray
        block.copyInto(array, startOffset)
    }
}
