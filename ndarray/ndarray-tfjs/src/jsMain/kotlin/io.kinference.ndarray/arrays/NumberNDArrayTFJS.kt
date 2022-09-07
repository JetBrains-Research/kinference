package io.kinference.ndarray.arrays

import io.kinference.ndarray.applyIf
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.PrimitiveType
import kotlin.coroutines.CoroutineContext

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

    override fun reshapeView(newShape: IntArray): NDArray {
        TODO("Not yet implemented")
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

    override fun concatenate(others: List<NDArray>, axis: Int): MutableNumberNDArrayTFJS {
        val otherArrays = Array(others.size) { (others[it] as NumberNDArrayTFJS).tfjsArray }
        val result = tfjsArray.concat(*otherArrays, axis = axis)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun tile(repeats: IntArray): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.tile(repeats.toTypedArray()))
    }

    override fun transpose2D(): NumberNDArrayTFJS {
        val newShape = tfjsArray.shape.reversedArray()
        return NumberNDArrayTFJS(tfjsArray.transpose(newShape))
    }

    override fun toMutable(newStrides: Strides): MutableNumberNDArrayTFJS {
        val tensor = tfjsArray.clone().applyIf(strides != newStrides) { it.reshape(newStrides.shape) }
        return MutableNumberNDArrayTFJS(tensor)
    }

    override fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNumberNDArray {
        TODO("Not yet implemented")
    }

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun row(row: Int): MutableNumberNDArray {
        TODO("Not yet implemented")
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return MutableNumberNDArrayTFJS(result)
    }

    override fun min(): Number {
        TODO("Not yet implemented")
    }

    override fun min(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
        val mins = tfjsArray.min(axis, keepDims)
        return NumberNDArrayTFJS(mins)
    }

    override fun max(): Number {
        TODO("Not yet implemented")
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

    override fun plus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.plus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun minus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.minus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun times(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.times(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun div(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.div(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun dot(other: NumberNDArray, destination: MutableNumberNDArray, coroutineContext: CoroutineContext): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun gemm(
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        lda: Int,
        b: NDArray,
        ldb: Int,
        beta: Double,
        c: MutableNDArray,
        ldc: Int,
        aOffset: Int,
        bOffset: Int,
        cOffset: Int,
        transposeA: Boolean,
        transposeB: Boolean
    ): MutableNumberNDArrayTFJS {
        TODO("Not yet implemented")
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

    override fun view(vararg axes: Int): NumberNDArrayTFJS {
        TODO("Not yet implemented")
    }

    override fun transpose(permutations: IntArray): NumberNDArrayTFJS {
        val result = tfjsArray.transpose(permutations.toTypedArray())
        return NumberNDArrayTFJS(result)
    }
}
