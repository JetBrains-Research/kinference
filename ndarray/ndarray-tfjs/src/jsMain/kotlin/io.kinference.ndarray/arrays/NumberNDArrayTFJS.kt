package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*

open class NumberNDArrayTFJS internal constructor(tfjsArray: ArrayTFJS) : NDArrayTFJS(tfjsArray), NumberNDArray {
    override fun get(index: IntArray): Number {
        return tfjsArray.bufferSync().get(*index) as Number
    }

    override fun getLinear(index: Int): Number {
        return tfjsArray.dataSync()[index] as Number
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

    override suspend fun expand(shape: IntArray): MutableNumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.broadcastTo(shape.toTypedArray()))
    }

    override fun toMutable(): MutableNumberNDArrayTFJS {
        val tensor = tfjsArray.clone()
        return MutableNumberNDArrayTFJS(tensor)
    }

    override suspend fun min(): Number {
        return tfjsArray.min().dataSync()[0] as Number
    }

    override suspend fun min(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
        val mins = tfjsArray.min(axis, keepDims)
        return NumberNDArrayTFJS(mins)
    }

    fun min(axes: Array<Int>, keepDims: Boolean): NumberNDArrayTFJS {
        val mins = tfjsArray.min(axes, keepDims)
        return NumberNDArrayTFJS(mins)
    }

    override suspend fun max(): Number {
        return tfjsArray.max().dataSync()[0] as Number
    }

    override suspend fun max(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
        val max = tfjsArray.max(axis, keepDims)
        return NumberNDArrayTFJS(max)
    }

    fun max(axes: Array<Int>, keepDims: Boolean): NumberNDArrayTFJS {
        val max = tfjsArray.max(axes, keepDims)
        return NumberNDArrayTFJS(max)
    }

    override suspend fun sum(): Number {
        return tfjsArray.sum().dataSync()[0] as Number
    }

    override suspend fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArrayTFJS {
        val result = tfjsArray.cumsum(axis, exclusive, reverse)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun erf(): NumberNDArrayTFJS {
        return NumberNDArrayTFJS(tfjsArray.erf())
    }

    private suspend fun applyMatrixLike(axis: Int, func: suspend NumberNDArrayTFJS.() -> NumberNDArrayTFJS): NumberNDArrayTFJS {
        return tidyNDArray {
            val rows = this.computeBlockSize(toDim = axis)
            val columns = this.computeBlockSize(fromDim = axis)
            val matrixShape = intArrayOf(rows,columns)

            val matrixLike = this.reshape(matrixShape).func()
            matrixLike.reshape(this.shape)
        }
    }

    private suspend fun softmaxNonLastAxis(axis: Int): NumberNDArrayTFJS {
        return applyMatrixLike(axis) { this.softmax(axis = -1) }
    }

    override suspend fun softmax(axis: Int): NumberNDArrayTFJS {
        val actualAxis = indexAxis(axis)

        return if (actualAxis == shape.lastIndex) {
            NumberNDArrayTFJS(tfjsArray.softmax(actualAxis))
        } else {
            softmaxNonLastAxis(actualAxis)
        }
    }

    private suspend fun logSoftmaxNonLastAxis(axis: Int): NumberNDArrayTFJS {
        return applyMatrixLike(axis) { this.logSoftmax(axis = -1) }
    }

    override suspend fun logSoftmax(axis: Int): NumberNDArrayTFJS {
        val actualAxis = indexAxis(axis)

        return if (actualAxis == shape.lastIndex) {
            NumberNDArrayTFJS(tfjsArray.logSoftmax(actualAxis))
        } else {
            logSoftmaxNonLastAxis(actualAxis)
        }
    }

    override suspend fun plus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.plus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun minus(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.minus(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun times(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.times(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun div(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.div(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun dot(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.dot(other.tfjsArray)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun argmax(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): NumberNDArrayTFJS {
        val actualAxis = indexAxis(axis)
        val result = tidy {
            val output = if (selectLastIndex) {
                val reversedInput = tfjsArray.reverse(actualAxis)
                val argMaxResult = reversedInput.argmax(actualAxis)
                val axisDimension = intScalar(this.shape[actualAxis] - 1).tfjsArray
                axisDimension - argMaxResult
            } else {
                tfjsArray.argmax(actualAxis)
            }

            arrayOf(
                if (keepDims) {
                    output.reshape(this.shape.copyOf().apply { set(actualAxis, 1) })
                } else {
                    output
                }
            )
        }.first()

        return NumberNDArrayTFJS(result)
    }

    override suspend fun argmin(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): NumberNDArrayTFJS {
        val actualAxis = indexAxis(axis)
        val result = tidy {
            val output = if (selectLastIndex) {
                val reversedInput = tfjsArray.reverse(actualAxis)
                val argMinResult = reversedInput.argmin(actualAxis)
                val axisDimension = intScalar(this.shape[actualAxis] - 1).tfjsArray
                axisDimension - argMinResult
            } else {
                tfjsArray.argmin(actualAxis)
            }

            arrayOf(
                if (keepDims) {
                    output.reshape(this.shape.copyOf().apply { set(actualAxis, 1) })
                } else {
                    output
                }
            )
        }.first()

        return NumberNDArrayTFJS(result)
    }

    override suspend fun reduceSum(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
        val sum = tfjsArray.sum(axes.toTypedArray(), keepDims)
        return NumberNDArrayTFJS(sum)
    }

    override suspend fun topK(axis: Int, k: Int, largest: Boolean, sorted: Boolean): Pair<NumberNDArrayTFJS, NumberNDArrayTFJS> {
        error("Operation topK() is not supported yet")
    }

    override suspend fun reshape(strides: Strides): NumberNDArrayTFJS {
        val result = tfjsArray.reshape(strides.shape)
        return NumberNDArrayTFJS(result)
    }

    override suspend fun reshape(shape: IntArray): NumberNDArrayTFJS {
        return reshape(Strides(shape))
    }

    override suspend fun transpose(permutations: IntArray): NumberNDArrayTFJS {
        val result = tfjsArray.transpose(permutations.toTypedArray())
        return NumberNDArrayTFJS(result)
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): NumberNDArrayTFJS {
        return super.pad(pads, mode, constantValue) as NumberNDArrayTFJS
    }

    override suspend fun matmul(other: NumberNDArray): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.matMul(other.tfjsArray, transposeLeft = false, transposeRight = false)
        return MutableNumberNDArrayTFJS(result)
    }

    fun matmul(
        other: NumberNDArray,
        transposeLeft: Boolean = false,
        transposeRight: Boolean = false,
    ): MutableNumberNDArrayTFJS {
        other as NumberNDArrayTFJS
        val result = tfjsArray.matMul(other.tfjsArray, transposeLeft, transposeRight)
        return MutableNumberNDArrayTFJS(result)
    }

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArrayTFJS {
        return MutableNumberNDArrayTFJS(tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray()))
    }

    override fun view(vararg axes: Int): NumberNDArrayTFJS {
        val indices = tensor(axes, arrayOf(axes.size), "int32")
        return NumberNDArrayTFJS(tfjsArray.gatherNd(indices)).also { indices.dispose() }
    }

    override suspend fun abs(): NumberNDArrayTFJS = NumberNDArrayTFJS(tfjsArray.abs())

    override fun asMutable() = MutableNumberNDArrayTFJS(tfjsArray)
}
