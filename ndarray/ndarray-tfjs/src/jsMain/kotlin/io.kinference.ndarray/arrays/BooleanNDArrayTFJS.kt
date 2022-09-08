package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.core.logicalNot
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

    override fun row(row: Int): MutableNDArray {
        TODO("Not yet implemented")
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

    override fun concatenate(others: List<NDArray>, axis: Int): MutableBooleanNDArrayTFJS {
        val otherArrays = Array(others.size) { (others[it] as BooleanNDArrayTFJS).tfjsArray }
        val result = tfjsArray.concat(*otherArrays, axis = axis)
        return MutableBooleanNDArrayTFJS(result)
    }

    override fun tile(repeats: IntArray): BooleanNDArrayTFJS {
        return BooleanNDArrayTFJS(tfjsArray.tile(repeats.toTypedArray()))
    }

    override fun transpose(permutations: IntArray): BooleanNDArrayTFJS {
        val result = tfjsArray.transpose(strides.shape.toTypedArray())
        return BooleanNDArrayTFJS(result)
    }

    override fun transpose2D(): BooleanNDArrayTFJS {
        val newShape = tfjsArray.shape.reversedArray()
        return BooleanNDArrayTFJS(tfjsArray.transpose(newShape))
    }

    fun not(): BooleanNDArrayTFJS {
        return BooleanNDArrayTFJS(logicalNot(tfjsArray))
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

    override fun clean() {
        val zerosArray = tensor(Array(linearSize) { false }, shapeArray, "bool")
        zerosArray.dispose()
        tfjsArray = zerosArray
    }
}
