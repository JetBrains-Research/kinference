package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.PrimitiveType

class MutableNumberNDArrayTFJS(tfjsArray: ArrayTFJS) : NumberNDArrayTFJS(tfjsArray), MutableNumberNDArray {
    override fun set(index: IntArray, value: Any) {
        require(value is PrimitiveType)
        tfjsArray.bufferSync().set(value, *index)
    }

    override fun clone(): MutableNumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.clone())
    }

    override fun clean() {
        val zerosArray = zeros(shapeArray, dtype)
        tfjsArray.dispose()
        tfjsArray = zerosArray
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as MutableNumberNDArrayTFJS
        val buffer = tfjsArray.bufferSync()
        val otherData = other.tfjsArray.dataSync()
        val startIndex = strides.index(offset)
        val iterator = NDIndexIterator(strides, from = startIndex)
        for (i in startInOther until endInOther) {
            buffer.set(otherData[i], *iterator.next())
        }
    }

    override fun fill(value: Any, from: Int, to: Int) {
        val offsetFrom = strides.index(from)
        val offsetTo = strides.index(to)
        val buffer = tfjsArray.bufferSync()
        ndIndexed(offsetFrom, offsetTo) { buffer.set(value, *it) }
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        val value = (array as NDArrayTFJS).tfjsArray.dataSync()[index]
        fill(value as Number, from, to)
    }

    override fun minusAssign(other: NumberNDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.minus(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun plusAssign(other: NumberNDArray) {
        val otherTFJS = (other as NDArrayTFJS).tfjsArray
        val result = tfjsArray.plus(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun timesAssign(other: NumberNDArray) {
        val otherTFJS = (other as NumberNDArrayTFJS).tfjsArray
        val result = tfjsArray.times(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }

    override fun divAssign(other: NumberNDArray) {
        val otherTFJS = (other as NumberNDArrayTFJS).tfjsArray
        val result = tfjsArray.div(otherTFJS)
        tfjsArray.dispose()
        tfjsArray = result
    }
}

