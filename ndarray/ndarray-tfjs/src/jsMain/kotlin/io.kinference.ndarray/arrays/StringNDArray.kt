package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

open class StringNDArrayTFJS(tfjsArray: ArrayTFJS) : NDArrayTFJS(tfjsArray) {
    override val type: DataType = DataType.ALL

    override fun get(index: IntArray): String {
        val value = tfjsArray.bufferSync().get(*index)
        return value as String
    }

    override fun getLinear(index: Int): String {
        val value = tfjsArray.bufferSync().get(index)
        return value as String
    }

    override fun singleValue(): String {
        require(this.linearSize == 1) { "NDArrays has more than 1 value" }
        val value = tfjsArray.dataSync()[0]
        return value as String
    }

    override suspend fun reshape(strides: Strides): StringNDArrayTFJS {
        val result = tfjsArray.reshape(strides.shape.toTypedArray())
        return StringNDArrayTFJS(result)
    }

    override fun toMutable(): MutableStringNDArrayTFJS {
        val tensor = tfjsArray.clone()
        return MutableStringNDArrayTFJS(tensor)
    }

    override fun copyIfNotMutable(): MutableStringNDArrayTFJS {
        return this as? MutableStringNDArrayTFJS ?: MutableStringNDArrayTFJS(tfjsArray.clone())
    }

    override fun clone(): StringNDArrayTFJS {
        return StringNDArrayTFJS(tfjsArray.clone())
    }

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableStringNDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return MutableStringNDArrayTFJS(result)
    }

    override suspend fun expand(shape: IntArray): MutableStringNDArrayTFJS {
        return MutableStringNDArrayTFJS(tfjsArray.broadcastTo(shape.toTypedArray()))
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): StringNDArrayTFJS {
        return super.pad(pads, mode, constantValue) as StringNDArrayTFJS
    }

    override fun view(vararg axes: Int): StringNDArrayTFJS {
        val indices = tensor(axes, arrayOf(axes.size), "int32")
        return StringNDArrayTFJS(tfjsArray.gatherNd(indices)).also { indices.dispose() }
    }
}

class MutableStringNDArrayTFJS(tfjsArray: ArrayTFJS) : StringNDArrayTFJS(tfjsArray), MutableNDArray {
    override fun clone(): MutableStringNDArrayTFJS {
        return MutableStringNDArrayTFJS(tfjsArray.clone())
    }
    override fun set(index: IntArray, value: Any) {
        require(value is String)
        tfjsArray.bufferSync().set(value, *index)
    }

    override fun setLinear(index: Int, value: Any) {
        require(value is String)
        tfjsArray.bufferSync().set(value, *strides.index(index))
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as MutableStringNDArrayTFJS
        val buffer = tfjsArray.bufferSync()
        val otherData = other.tfjsArray.dataSync()
        val startIndex = strides.index(offset)
        val iterator = NDIndexer(strides, from = startIndex)
        for (i in startInOther until endInOther) {
            buffer.set(otherData[i], *iterator.next())
        }
    }

    override fun fill(value: Any, from: Int, to: Int) {
        require(value is String)
        val offsetFrom = strides.index(from)
        val offsetTo = strides.index(to)
        val buffer = tfjsArray.bufferSync()
        ndIndices(offsetFrom, offsetTo) { buffer.set(value, *it) }
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        val value = (array as NDArrayTFJS).tfjsArray.dataSync()[index]
        fill(value as String, from, to)
    }

    override fun clean() {
        val zerosArray = tensor(Array(linearSize) { false }, shapeArray, "bool")
        zerosArray.dispose()
        tfjsArray = zerosArray
    }

    override fun viewMutable(vararg axes: Int): MutableStringNDArrayTFJS {
        val indices = tensor(axes, arrayOf(axes.size), "int32")
        return MutableStringNDArrayTFJS(tfjsArray.gatherNd(indices)).also { indices.dispose() }
    }
}
