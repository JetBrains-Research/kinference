package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.clone
import io.kinference.primitives.types.PrimitiveType

class MutableNumberNDArrayTFJS(tfjsArray: ArrayTFJS) : NumberNDArrayTFJS(tfjsArray), MutableNumberNDArray {
    override fun set(index: IntArray, value: Any) {
        require(value is PrimitiveType)
        tfjsArray.bufferSync().set(value, *index)
    }

    override fun clone(): NumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.clone())
    }

    override fun clean() {
        TODO("Not yet implemented")
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        TODO("Not yet implemented")
    }

    override fun divAssign(other: NDArray) {
        TODO("Not yet implemented")
    }

    override fun fill(value: Any, from: Int, to: Int) {
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        TODO("Not yet implemented")
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray {
        TODO("Not yet implemented")
    }

    override fun minusAssign(other: NDArray) {
        TODO("Not yet implemented")
    }

    override fun plusAssign(other: NDArray) {
        TODO("Not yet implemented")
    }

    override fun timesAssign(other: NDArray) {
        TODO("Not yet implemented")
    }

    override fun viewMutable(vararg axes: Int): MutableNumberNDArray {
        TODO("Not yet implemented")
    }
}

