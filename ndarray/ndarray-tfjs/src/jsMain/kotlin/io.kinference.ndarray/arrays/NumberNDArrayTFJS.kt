package io.kinference.ndarray.arrays

import io.kinference.ndarray.applyIf
import io.kinference.ndarray.extensions.*
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext

open class NumberNDArrayTFJS(tfjsArray: ArrayTFJS) : NDArrayTFJS(tfjsArray), NumberNDArray {
    override fun get(index: IntArray): Number {
        return tfjsArray.bufferSync().get(*index) as Number
    }

    override fun singleValue(): Number {
        require(this.linearSize == 1) { "NDArrays has more than 1 value" }
        return tfjsArray.dataSync()[0] as Number
    }

    override fun copyIfNotMutable(): MutableNumberNDArrayTFJS {
        return this as? MutableNumberNDArrayTFJS ?: MutableNumberNDArrayTFJS(tfjsArray.clone())
    }

    override fun clone(): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.clone())
    }

    override fun expand(shape: IntArray): MutableNumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.broadcastTo(shape.toTypedArray()))
    }

    override fun pad(pads: Array<Pair<Int, Int>>, mode: String, constantValue: NDArray?): NumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun nonZero(): NumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun toMutable(): MutableNumberNDArrayTFJS {
        val tensor = tfjsArray.clone()
        return MutableNumberNDArrayTFJS(tensor)
    }

    override fun min(): Number {
        return tfjsArray.min().dataSync()[0] as Number
    }

    override fun min(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
        val mins = tfjsArray.min(axis, keepDims)
        return NumberNDArrayTFJS(mins)
    }

    override fun max(): Number {
        return tfjsArray.max().dataSync()[0] as Number
    }

    override fun max(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
        val max = tfjsArray.max(axis, keepDims)
        return NumberNDArrayTFJS(max)
    }

    override fun sum(): Number {
        return tfjsArray.sum().dataSync()[0] as Number
    }

    override fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArrayTFJS {
        val result = tfjsArray.cumsum(axis, exclusive, reverse)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun erf(): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.erf())
    }

    override fun softmax(axis: Int, coroutineContext: CoroutineContext?): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.softmax(axis))
    }

    override fun logSoftmax(axis: Int, coroutineContext: CoroutineContext?): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.logSoftmax(axis))
    }

    override fun plus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.plus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun minus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.minus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun times(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.times(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun div(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.div(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun dot(other: NumberNDArray, coroutineContext: CoroutineContext): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.dot(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun argmax(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): NumberNDArrayTFJS {
        val result = tfjsArray.argmax(axis)
        return NumberNDArrayTFJS(result)
    }

    override fun reduceSum(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
        val sum = tfjsArray.sum(axes.toTypedArray(), keepDims)
        return NumberNDArrayTFJS(sum)
    }

    override fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<NumberNDArrayTFJS, NumberNDArrayTFJS> {
        TODO("Not yet implemented")
    }

    override fun reshape(strides: Strides): NumberNDArrayTFJS {
        val result = tfjsArray.reshape(strides.shape)
        return NumberNDArrayTFJS(result)
    }

    override fun reshape(shape: IntArray): NumberNDArrayTFJS {
        return reshape(Strides(shape))
    }

    override fun transpose(permutations: IntArray): NumberNDArrayTFJS {
        val result = tfjsArray.transpose(permutations.toTypedArray())
        return NumberNDArrayTFJS(result)
    }

    override fun matmul(other: NumberNDArray, coroutineContext: CoroutineContext): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.matMul(other.tfjsArray, transposeLeft = false, transposeRight = false)
        return MutableNumberNDArrayTFJS(result)
    }

    fun matmul(
        other: NumberNDArray,
        transposeLeft: Boolean = false,
        transposeRight: Boolean = false,
        coroutineContext: CoroutineContext = EmptyCoroutineContext
    ): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.matMul(other.tfjsArray, transposeLeft, transposeRight)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray()))
    }
}
