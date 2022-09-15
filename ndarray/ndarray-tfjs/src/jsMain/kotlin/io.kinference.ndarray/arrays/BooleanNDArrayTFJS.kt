package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

open class BooleanNDArrayTFJS(tfjsArray: ArrayTFJS) : NDArrayTFJS(tfjsArray) {
    override val type: DataType = DataType.BOOLEAN

    override fun get(index: IntArray): Boolean {
        return tfjsArray.bufferSync().get(*index) as Boolean
    }

    override fun singleValue(): Boolean {
        require(this.linearSize == 1) { "NDArrays has more than 1 value" }
        return tfjsArray.dataSync()[0] as Boolean
    }

    override fun reshape(strides: Strides): BooleanNDArrayTFJS {
        val result = tfjsArray.reshape(strides.shape.toTypedArray())
        return BooleanNDArrayTFJS(result)
    }

    override fun toMutable(newStrides: Strides): MutableBooleanNDArrayTFJS {
        val tensor = tfjsArray.clone().applyIf(strides != newStrides) { it.reshape(newStrides.shape) }
        return MutableBooleanNDArrayTFJS(tensor)
    }

    override fun copyIfNotMutable(): MutableBooleanNDArrayTFJS{
        return this as? MutableBooleanNDArrayTFJS ?: MutableBooleanNDArrayTFJS(tfjsArray.clone())
    }

    override fun clone(): BooleanNDArrayTFJS {
        return BooleanNDArrayTFJS(tfjsArray.clone())
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableBooleanNDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return MutableBooleanNDArrayTFJS(result)
    }

    override fun expand(shape: IntArray): MutableBooleanNDArrayTFJS {
        return MutableBooleanNDArrayTFJS(tfjsArray.broadcastTo(shape.toTypedArray()))
    }

    override fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): MutableBooleanNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun nonZero(): NumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    fun not(): BooleanNDArrayTFJS {
        return BooleanNDArrayTFJS(tfjsArray.not())
    }
}

class MutableBooleanNDArrayTFJS(tfjsArray: ArrayTFJS) : BooleanNDArrayTFJS(tfjsArray), MutableNDArray {
    override fun clone(): MutableBooleanNDArrayTFJS {
        return MutableBooleanNDArrayTFJS(tfjsArray.clone())
    }
    override fun set(index: IntArray, value: Any) {
        require(value is Boolean)
        tfjsArray.bufferSync().set(value, *index)
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as MutableBooleanNDArrayTFJS
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

    override fun clean() {
        val zerosArray = tensor(Array(linearSize) { false }, shapeArray, "bool")
        zerosArray.dispose()
        tfjsArray = zerosArray
    }
}
