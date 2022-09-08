package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*
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
        val zerosArray = zeros(shapeArray, dtype)
        tfjsArray.dispose()
        tfjsArray = zerosArray
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        TODO("Not yet implemented")
    }

    override fun fill(value: Any, from: Int, to: Int) {
        val offsetFrom = strides.index(from)
        val offsetTo = strides.index(to)
        val buffer = tfjsArray.bufferSync()
        ndIndexed(offsetFrom, offsetTo) { buffer.set(value, *it) }
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        TODO("Not yet implemented")
    }

    override fun minusAssign(other: NDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.minus(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun plusAssign(other: NDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.plus(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun timesAssign(other: NDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.times(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun divAssign(other: NDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.div(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }
}

