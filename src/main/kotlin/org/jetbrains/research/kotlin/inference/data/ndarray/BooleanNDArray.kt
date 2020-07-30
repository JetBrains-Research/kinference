package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.onnx.TensorProto

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

    override fun plus(other: NDArray<BooleanArray>, copy: Boolean): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun times(other: NDArray<BooleanArray>, copy: Boolean): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun div(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun minus(other: NDArray<BooleanArray>): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun mapElements(func: (Any) -> Any, copy: Boolean): NDArray<BooleanArray> {
        TODO("Not yet implemented")
    }

    override fun clean() {
        TODO("Not yet implemented")
    }

    override fun slice(sliceLength: Int, start: Int): BooleanArray {
        return array.sliceArray(start until start + sliceLength)
    }

    override fun placeAll(startOffset: Int, block: Any?) {
        block as BooleanArray
        block.copyInto(array, startOffset)
    }
}
