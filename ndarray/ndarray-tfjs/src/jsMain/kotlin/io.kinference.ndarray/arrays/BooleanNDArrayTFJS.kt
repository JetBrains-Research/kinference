package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

open class BooleanNDArrayTFJS(tfjsArray: ArrayTFJS) : NDArrayTFJS(tfjsArray) {
    override val type: DataType = DataType.BOOLEAN

    override fun get(index: IntArray): Boolean {
        val value = tfjsArray.bufferSync().get(*index)
        return value != 0
    }

    override fun getLinear(index: Int): Boolean {
        val value = tfjsArray.dataSync()[index]
        return value != 0
    }

    override fun singleValue(): Boolean {
        require(this.linearSize == 1) { "NDArrays has more than 1 value" }
        val value = tfjsArray.dataSync()[0]
        return value != 0
    }

    override suspend fun reshape(strides: Strides): BooleanNDArrayTFJS {
        val result = tfjsArray.reshape(strides.shape.toTypedArray())
        return BooleanNDArrayTFJS(result)
    }

    override fun toMutable(): MutableBooleanNDArrayTFJS {
        val tensor = tfjsArray.clone()
        return MutableBooleanNDArrayTFJS(tensor)
    }

    override fun copyIfNotMutable(): MutableBooleanNDArrayTFJS{
        return this as? MutableBooleanNDArrayTFJS ?: MutableBooleanNDArrayTFJS(tfjsArray.clone())
    }

    override fun clone(): BooleanNDArrayTFJS {
        return BooleanNDArrayTFJS(tfjsArray.clone())
    }

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableBooleanNDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return MutableBooleanNDArrayTFJS(result)
    }

    override suspend fun expand(shape: IntArray): MutableBooleanNDArrayTFJS {
        return MutableBooleanNDArrayTFJS(tfjsArray.broadcastTo(shape.toTypedArray()))
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): BooleanNDArrayTFJS {
        require(mode == PadMode.CONSTANT) { "Only CONSTANT pad mode is supported for TFJS backend" }
        require(constantValue == null || constantValue is NDArrayTFJS)
        val padsArray = Array(pads.size) { arrayOf(pads[it].first, pads[it].second) }
        return tidyNDArray {
            val value = constantValue as? NDArrayTFJS ?: zero(dtype)
            return@tidyNDArray BooleanNDArrayTFJS(tfjsArray.pad(padsArray, value.singleValue()))
        }
    }

    override fun view(vararg axes: Int): BooleanNDArrayTFJS {
        val indices = tensor(axes, arrayOf(axes.size), "int32")
        return BooleanNDArrayTFJS(tfjsArray.gatherNd(indices)).also { indices.dispose() }
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

    override fun setLinear(index: Int, value: Any) {
        require(value is Boolean)
        tfjsArray.bufferSync().set(value, *strides.index(index))
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as MutableBooleanNDArrayTFJS
        val buffer = tfjsArray.bufferSync()
        val otherData = other.tfjsArray.dataSync()
        val startIndex = strides.index(offset)
        val iterator = NDIndexer(strides, from = startIndex)
        for (i in startInOther until endInOther) {
            buffer.set(otherData[i], *iterator.next())
        }
    }

    override fun fill(value: Any, from: Int, to: Int) {
        val offsetFrom = strides.index(from)
        val offsetTo = strides.index(to)
        val buffer = tfjsArray.bufferSync()
        ndIndices(offsetFrom, offsetTo) { buffer.set(value, *it) }
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

    override fun viewMutable(vararg axes: Int): MutableBooleanNDArrayTFJS {
        val indices = tensor(axes, arrayOf(axes.size), "int32")
        return MutableBooleanNDArrayTFJS(tfjsArray.gatherNd(indices)).also { indices.dispose() }
    }
}
