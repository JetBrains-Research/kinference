@file:Suppress("UNCHECKED_CAST")

package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.core.*
import io.kinference.primitives.types.DataType
import io.kinference.utils.toTypedIntArray
import io.kinference.ndarray.resolveTFJSDataType
import io.kinference.utils.toIntArray

internal val NDArrayTFJS.dtype: String
    get() = tfjsArray.dtype

val NDArrayTFJS.shapeArray: Array<Int>
    get() = tfjsArray.shape

internal fun ArrayTFJS.toNDArray() = makeNDArray(this, dtype)

internal fun Array<out NDArrayTFJS>.getArrays() = Array(this.size) { this[it].tfjsArray }
internal fun List<NDArrayTFJS>.getArrays() = Array(this.size) { this[it].tfjsArray }

internal fun Array<out ArrayTFJS>.getNDArrays() = Array(this.size) { this[it].toNDArray() }

fun <T : NDArrayTFJS> T.dataInt() = tfjsArray.dataInt()
fun <T : NDArrayTFJS> T.dataFloat() = tfjsArray.dataFloat()
fun <T : NDArrayTFJS> T.dataBool() = tfjsArray.dataBool()
fun <T : NDArrayTFJS> T.dataString() = tfjsArray.dataString()

fun <T : NDArrayTFJS> T.broadcastTo(shape: Array<Int>) = tfjsArray.broadcastTo(shape).toNDArray() as T

fun <T : NDArrayTFJS> T.castToInt(): NumberNDArrayTFJS = (if (type == DataType.INT) this.clone() else tfjsArray.cast("int32").toNDArray()) as NumberNDArrayTFJS
fun <T : NDArrayTFJS> T.castToFloat(): NumberNDArrayTFJS = (if (type == DataType.FLOAT) this.clone() else tfjsArray.cast("float32").toNDArray()) as NumberNDArrayTFJS
fun <T : NDArrayTFJS> T.castToBool(): BooleanNDArrayTFJS = (if (type == DataType.BOOLEAN) this.clone() else tfjsArray.cast("bool").toNDArray()) as BooleanNDArrayTFJS

fun <T : NDArrayTFJS> Array<T>.concat(axis: Int = 0) = concat(getArrays(), axis).toNDArray() as T

fun <T : NDArrayTFJS> Collection<T>.concat(axis: Int = 0) = concat(toTypedArray().getArrays(), axis).toNDArray() as T

fun <T : NDArrayTFJS> T.transpose() = transpose(tfjsArray, null).toNDArray() as T

fun <T : NDArrayTFJS> T.unstack(axis: Int = 0) = unstack(tfjsArray, axis).getNDArrays() as Array<T>

fun <T : NDArrayTFJS> Array<T>.stack(axis: Int = 0) = stack(getArrays(), axis).toNDArray() as T

fun <T : NDArrayTFJS> Collection<T>.stack(axis: Int = 0) = this.toTypedArray().stack(axis)

fun <T : NDArrayTFJS> T.flatten() = reshape(tfjsArray, arrayOf(this.linearSize)).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse(axes: Array<Int>) = reverse(tfjsArray, axes).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse(axis: Int) = reverse(tfjsArray, arrayOf(axis)).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse() = reverse(tfjsArray, null).toNDArray() as T

fun <T : NDArrayTFJS> T.equal(other: NDArrayTFJS) = BooleanNDArrayTFJS(equal(tfjsArray, other.tfjsArray))

fun <T : NDArrayTFJS> T.notEqual(other: NDArrayTFJS) = BooleanNDArrayTFJS(notEqual(tfjsArray, other.tfjsArray))

fun <T : NDArrayTFJS> T.where(condition: NDArrayTFJS, other: NDArrayTFJS) = where(condition.tfjsArray, tfjsArray, other.tfjsArray).toNDArray()

fun <T : NDArrayTFJS> T.where() = whereAsync(tfjsArray).then { it.toNDArray() }

fun <T : NDArrayTFJS> T.pad(paddings: Array<Array<Int>>, constantValue: Number) = pad(tfjsArray, paddings, constantValue).toNDArray() as T

fun <T : NDArrayTFJS> T.gatherNd(indices: NDArrayTFJS) = gatherND(tfjsArray, indices.tfjsArray).toNDArray()

fun <T : NDArrayTFJS> T.topk(k: Int, sorted: Boolean = false) = topk(tfjsArray, k, sorted).let { it.first.toNDArray() to it.second.toNDArray() }

fun NumberNDArrayTFJS.leakyRelu(alpha: Number) = NumberNDArrayTFJS(leakyRelu(tfjsArray, alpha))

fun NumberNDArrayTFJS.sum(axis: Int, keepDims: Boolean = false) = NumberNDArrayTFJS(sum(tfjsArray, arrayOf(axis), keepDims))

fun NumberNDArrayTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = NumberNDArrayTFJS(sum(tfjsArray, axes, keepDims))

fun Array<NumberNDArrayTFJS>.sum() = NumberNDArrayTFJS(addN(getArrays()))

fun NumberNDArrayTFJS.add(other: NumberNDArrayTFJS) = NumberNDArrayTFJS(add(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.add(tensors: Array<NumberNDArrayTFJS>) = NumberNDArrayTFJS(addN(arrayOf(tfjsArray, *tensors.getArrays())))

fun NumberNDArrayTFJS.add(vararg tensors: NumberNDArrayTFJS) = add(tensors as Array<NumberNDArrayTFJS>)

fun NumberNDArrayTFJS.min(keepDims: Boolean = false) = NumberNDArrayTFJS(tfjsArray.min(keepDims))

fun NumberNDArrayTFJS.max(keepDims: Boolean = false) = NumberNDArrayTFJS(max(tfjsArray, null, keepDims))

fun min(a: NumberNDArrayTFJS, b: NumberNDArrayTFJS) = NumberNDArrayTFJS(minimum(a.tfjsArray, b.tfjsArray))

fun max(a: NumberNDArrayTFJS, b: NumberNDArrayTFJS) = NumberNDArrayTFJS(maximum(a.tfjsArray, b.tfjsArray))

fun NumberNDArrayTFJS.round() = NumberNDArrayTFJS(round(tfjsArray))

fun NumberNDArrayTFJS.clip(minValue: Number, maxValue: Number) = NumberNDArrayTFJS(clipByValue(tfjsArray, minValue, maxValue))

operator fun NumberNDArrayTFJS.unaryMinus() = NumberNDArrayTFJS(neg(tfjsArray))

fun NumberNDArrayTFJS.sqrt() = NumberNDArrayTFJS(sqrt(tfjsArray))

fun sqrt(value: NumberNDArrayTFJS) = value.sqrt()

fun NumberNDArrayTFJS.tanh() = NumberNDArrayTFJS(tanh(tfjsArray))

fun NumberNDArrayTFJS.acosh() = NumberNDArrayTFJS(tfjsArray.acosh())

fun NumberNDArrayTFJS.relu() = NumberNDArrayTFJS(tfjsArray.relu())

fun relu(value: NumberNDArrayTFJS) = value.relu()

fun NumberNDArrayTFJS.less(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(less(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.greater(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(greater(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.greaterEqual(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(greaterEqual(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.log() = NumberNDArrayTFJS(tfjsArray.log())

fun NumberNDArrayTFJS.sigmoid() = NumberNDArrayTFJS(sigmoid(tfjsArray))

fun NumberNDArrayTFJS.acos() = NumberNDArrayTFJS(tfjsArray.acos())

fun NumberNDArrayTFJS.asin() = NumberNDArrayTFJS(tfjsArray.asin())

fun NumberNDArrayTFJS.asinh() = NumberNDArrayTFJS(tfjsArray.asinh())

fun NumberNDArrayTFJS.sinh() = NumberNDArrayTFJS(tfjsArray.sinh())

fun NumberNDArrayTFJS.atan() = NumberNDArrayTFJS(tfjsArray.atan())

fun NumberNDArrayTFJS.atanh() = NumberNDArrayTFJS(tfjsArray.atanh())

fun NumberNDArrayTFJS.tan() = NumberNDArrayTFJS(tfjsArray.tan())

fun NumberNDArrayTFJS.sin() = NumberNDArrayTFJS(tfjsArray.sin())

fun NumberNDArrayTFJS.moments(axis: Int, keepDims: Boolean = false) = tfjsArray.moments(axis, keepDims).toNDArray()

fun NumberNDArrayTFJS.moments(axes: Array<Int>, keepDims: Boolean = false) = tfjsArray.moments(axes, keepDims).toNDArray()

fun <T : NDArrayTFJS> T.tensorScatterUpdate(indices: NDArrayTFJS, updates: NDArrayTFJS): T {
    return tfjsArray.tensorScatterUpdate(indices.tfjsArray, updates.tfjsArray).toNDArray() as T
}

fun NumberNDArrayTFJS.ceil() = NumberNDArrayTFJS(tfjsArray.ceil())

fun NumberNDArrayTFJS.exp() = NumberNDArrayTFJS(tfjsArray.exp())

fun NumberNDArrayTFJS.expm1() = NumberNDArrayTFJS(tfjsArray.expm1())

fun NumberNDArrayTFJS.elu() = NumberNDArrayTFJS(tfjsArray.elu())

fun NumberNDArrayTFJS.prelu(alpha: NumberNDArrayTFJS) = NumberNDArrayTFJS(tfjsArray.prelu(alpha.tfjsArray))

fun NumberNDArrayTFJS.cos() = NumberNDArrayTFJS(tfjsArray.cos())

fun NumberNDArrayTFJS.cosh() = NumberNDArrayTFJS(tfjsArray.cosh())

fun NumberNDArrayTFJS.qrDecomposition(fullMatrices: Boolean = false) = tfjsArray.qrDecomposition(fullMatrices).toNDArray()

fun NumberNDArrayTFJS.prod(axis: Int, keepDims: Boolean = false) = NumberNDArrayTFJS(tfjsArray.prod(axis, keepDims))

fun NumberNDArrayTFJS.prod(axes: Array<Int>, keepDims: Boolean = false) = NumberNDArrayTFJS(tfjsArray.prod(axes, keepDims))

suspend fun NumberNDArrayTFJS.det(): NumberNDArrayTFJS {
    val result = tidyNDArray {
        val n = shape.last()
        val qrResult = qrDecomposition()

        val newShapeForR = intArrayOf(*shape.sliceArray(0 until shape.size - 2), n * n)
        val reshapedR = qrResult.r.reshape(newShapeForR)

        val indicesForGather = NDArrayTFJS.intRange(0, n * n, n + 1)
        val diagonalOfR = reshapedR.gather(indicesForGather, axis = -1) as NumberNDArrayTFJS

        return@tidyNDArray diagonalOfR.prod(axis = -1)
    }

    return result
}

fun NumberNDArrayTFJS.floor() = NumberNDArrayTFJS(tfjsArray.floor())

suspend fun NumberNDArrayTFJS.hardmax(axis: Int = 1): NumberNDArrayTFJS {
    val actualAxis = indexAxis(axis)
    val rows = computeBlockSize(toDim = actualAxis)
    val columns = computeBlockSize(fromDim = actualAxis)

    return tidyNDArray {
        val reshapedInput = this.reshape(intArrayOf(rows, columns))
        val argMaxOfInput = reshapedInput.argmax(axis = 1, keepDims = false)

        val output = oneHot(
            indices = argMaxOfInput.tfjsArray,
            depth = columns,
            onValue = 1,
            offValue = 0,
            dtype = this.tfjsArray.dtype
        ).toNDArray() as NumberNDArrayTFJS

        return@tidyNDArray output.reshape(this.strides)
    }
}

suspend fun NumberNDArrayTFJS.isInf(detectNegative: Boolean = true, detectPositive: Boolean = true): BooleanNDArrayTFJS = tidyNDArray {
    val infPositions = BooleanNDArrayTFJS(tfjsArray.isInf())

    return@tidyNDArray when {
        detectNegative && detectPositive -> infPositions

        detectNegative -> {
            val negativePositions = this.less(NDArrayTFJS.floatScalar(0f))

            infPositions.and(negativePositions)
        }

        detectPositive -> {
            val positivePositions = this.greater(NDArrayTFJS.floatScalar(0f))

            infPositions.and(positivePositions)
        }

        else -> error("At least one of detectNegative or detectPositive must be true")
    }
}

fun NumberNDArrayTFJS.isNaN() = BooleanNDArrayTFJS(tfjsArray.isNaN())

suspend fun List<NumberNDArrayTFJS>.max(): NumberNDArrayTFJS {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    return tidyNDArray { reduce { acc, next -> max(acc, next) } }
}

suspend fun Array<out NumberNDArrayTFJS>.max() = toList().max()

suspend fun maxOf(vararg inputs: NumberNDArrayTFJS) = inputs.max()

suspend fun List<NumberNDArrayTFJS>.min(): NumberNDArrayTFJS {
    require(isNotEmpty()) { "Input array must have at least one element" }
    if (size == 1) return single()

    return tidyNDArray { reduce { acc, next -> min(acc, next) } }
}

suspend fun Array<out NumberNDArrayTFJS>.min() = toList().min()

suspend fun minOf(vararg inputs: NumberNDArrayTFJS) = inputs.min()

suspend fun <T : NDArrayTFJS> T.trilu(k: Int = 0, upper: Boolean = true): T {
    require(rank >= 2) { "Input tensor has to be at least of rank=2, but tensor of rank=${rank} was found" }

    if (this.linearSize == 0) return this.clone() as T

    return tidyNDArray {
        if (upper) this.triluUpper(k) else this.triluLower(k)
    } as T
}

private fun NDArrayTFJS.triluUpper(k: Int): NDArrayTFJS {
    val (height, width) = shape.takeLast(2)

    if (k == 0) return tfjsArray.bandPart(numUpper = -1).toNDArray()
    if (k > 0 && k - 1 > width) return NDArrayTFJS.zerosOfType(shapeArray, dtype)
    if (k < 0 && -k > height) return this.clone()

    return if (k > 0) {
        tfjsArray - tfjsArray.bandPart(numLower = -1, numUpper = k - 1)
    } else {
        tfjsArray.bandPart(numLower = -k, numUpper = -1)
    }.toNDArray()
}

private fun NDArrayTFJS.triluLower(k: Int): NDArrayTFJS {
    val (height, width) = shape.takeLast(2)

    if (k == 0) return tfjsArray.bandPart(numLower = -1).toNDArray()
    if (k < 0 && -k - 1 > height) return NDArrayTFJS.zerosOfType(shapeArray, dtype)
    if (k > 0 && k > width) return this.clone()

    return if (k > 0) {
        tfjsArray.bandPart(numLower = -1, numUpper = k)
    } else {
        tfjsArray - tfjsArray.bandPart(numLower = -k - 1, numUpper = -1)
    }.toNDArray()
}

fun List<NumberNDArrayTFJS>.sum(): NumberNDArrayTFJS {
    if (isEmpty()) error("Array for sum operation must have at least one element")
    if (size == 1) return single()

    return getArrays().sum().toNDArray() as NumberNDArrayTFJS
}

fun Array<out NumberNDArrayTFJS>.sum() = this.toList().sum()

fun sumOf(vararg inputs: NumberNDArrayTFJS) = inputs.sum()

suspend fun List<NumberNDArrayTFJS>.mean(): NumberNDArrayTFJS {
    if (isEmpty()) error("Array for mean operation must have at least one element")
    if (size == 1) return single()

    return tidyNDArray { this.sum() / NDArrayTFJS.intScalar(size) }
}

suspend fun Array<out NumberNDArrayTFJS>.mean() = toList().mean()

suspend fun meanOf(vararg inputs: NumberNDArrayTFJS) = inputs.mean()

fun NumberNDArrayTFJS.sign() = NumberNDArrayTFJS(tfjsArray.sign())

fun NumberNDArrayTFJS.reduceMean(axis: Int, keepDims: Boolean): NumberNDArrayTFJS {
    val meanArray = tfjsArray.mean(axis, keepDims)
    return NumberNDArrayTFJS(meanArray)
}

fun NumberNDArrayTFJS.reduceMean(axes: Array<Int>, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    val meanArray = tfjsArray.mean(axes, keepDims)
    return NumberNDArrayTFJS(meanArray)
}

suspend fun NumberNDArrayTFJS.reduceL1(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    return tidyNDArray {
        val absTensor = this.abs()
        return@tidyNDArray absTensor.reduceSum(axes, keepDims)
    }
}

suspend fun NumberNDArrayTFJS.reduceL1(axis: Int, keepDims: Boolean): NumberNDArrayTFJS = reduceL1(intArrayOf(axis), keepDims)

suspend fun NumberNDArrayTFJS.reduceSumSquare(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    return tidyNDArray {
        val squaredArray = this * this
        return@tidyNDArray squaredArray.reduceSum(axes, keepDims)
    }
}

suspend fun NumberNDArrayTFJS.reduceSumSquare(axis: Int, keepDims: Boolean) = reduceSumSquare(intArrayOf(axis), keepDims)

suspend fun NumberNDArrayTFJS.reduceL2(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    return tidyNDArray {
        val squaredSum = reduceSumSquare(axes, keepDims)
        return@tidyNDArray squaredSum.sqrt()
    }
}

suspend fun NumberNDArrayTFJS.reduceL2(axis: Int, keepDims: Boolean) = reduceL2(intArrayOf(axis), keepDims)

suspend fun NumberNDArrayTFJS.reduceLogSum(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    return tidyNDArray {
        val sum = reduceSum(axes, keepDims)
        return@tidyNDArray sum.log()
    }
}

suspend fun NumberNDArrayTFJS.reduceLogSum(axis: Int, keepDims: Boolean) = reduceLogSum(intArrayOf(axis), keepDims)


suspend fun NumberNDArrayTFJS.reduceLogSumExp(axes: IntArray, keepDims: Boolean): NumberNDArrayTFJS {
    if (axes.isEmpty()) return this

    return tidyNDArray {
        val expTensor = this.exp()
        val sumTensor = expTensor.reduceSum(axes, keepDims)

        return@tidyNDArray sumTensor.log()
    }
}

suspend fun NumberNDArrayTFJS.reciprocal() = NumberNDArrayTFJS(tfjsArray.reciprocal())

suspend fun <T : NDArrayTFJS> T.reverseSeq(mode: ReverseSeqMode, seqLens: IntArray): T {
    require(dtype != "string") { "String tensors are not supported by current version of reverseSeq" }

    val numBatches = if (mode is ReverseSeqMode.BatchMajorMode) shape[0] else shape[1]
    val maxSeqLen = if (mode is ReverseSeqMode.BatchMajorMode) shape[1] else shape[0]

    require(seqLens.size == numBatches) { "Sequence lengths array size must have $numBatches elements but the array of size ${seqLens.size} was found" }

    val blockSize = computeBlockSize(fromDim = 2)
    val output = NDArrayTFJS.zerosOfType(shapeArray, dtype)

    val outputBuffer = output.tfjsArray.bufferSync()
    val inputData = this.tfjsArray.dataSync()

    for (batchIdx in 0 until numBatches) {
        val seqLength = seqLens[batchIdx]
        require(seqLength in 0..maxSeqLen) { "Sequence length must be in range $[0, $maxSeqLen], current seq length=$seqLength" }

        for (seqIdx in 0 until seqLength) {
            val inputOffset = mode.index(batchIdx, seqIdx, numBatches, maxSeqLen, blockSize)
            val outputOffset = mode.reverseIndex(batchIdx, seqIdx, seqLength, numBatches, maxSeqLen, blockSize)

            val startIndex = strides.index(outputOffset)
            val iterator = NDIndexer(strides, from = startIndex)
            for (i in inputOffset until inputOffset + blockSize) {
                outputBuffer.set(inputData[i], *iterator.next())
            }
        }

        for (seqIdx in seqLength until maxSeqLen) {
            val offset = mode.index(batchIdx, seqIdx, numBatches, maxSeqLen, blockSize)

            val startIndex = strides.index(offset)
            val iterator = NDIndexer(strides, from = startIndex)
            for (i in offset until offset + blockSize) {
                outputBuffer.set(inputData[i], *iterator.next())
            }
        }
    }
    return output.asMutable() as T
}

suspend fun NumberNDArrayTFJS.mod(other: NumberNDArrayTFJS) = NumberNDArrayTFJS(tfjsArray.mod(other.tfjsArray))

suspend fun NumberNDArrayTFJS.fmod(other: NumberNDArrayTFJS): NumberNDArrayTFJS {
    return tidyNDArray {
        val modResult = mod(other)
        val fmodResult = modResult - other

        val zero = NDArrayTFJS.intScalar(0)

        val leftLessZero = this.less(zero)
        val rightLessZero = other.less(zero)
        val modResultNotEqualsZero = modResult.notEqual(zero)

        val maskForFmod = (leftLessZero xor rightLessZero) and modResultNotEqualsZero

        return@tidyNDArray fmodResult.where(maskForFmod, modResult) as NumberNDArrayTFJS
    }
}

suspend operator fun NumberNDArrayTFJS.rem(other: NumberNDArrayTFJS) = fmod(other)

suspend fun NumberNDArrayTFJS.pow(exp: NumberNDArrayTFJS) = NumberNDArrayTFJS(tfjsArray.pow(exp.tfjsArray))

suspend fun NDArrayTFJS.Companion.oneHot(indices: NumberNDArrayTFJS, depth: Int, values: NDArrayTFJS, axis: Int = -1): NDArrayTFJS {
    fun NDArrayTFJS.parseValues(): List<Number> = when (values.tfjsArray.dtype) {
        "float32" -> values.dataFloat().take(2)
        "int32" -> values.dataInt().take(2)
        "bool" -> values.dataBool().take(2).map { it.toInt() }
        else -> error("Unsupported \"values\" data type: $type")
    }

    require(values.linearSize == 2 && values.rank == 1) {
        "\"values\" must be two-element array of format [off_value, on_value], current array rank=${indices.rank}, linearSize=${indices.linearSize}"
    }
    val (offValue, onValue) = values.parseValues()

    return oneHot(indices, depth, offValue, onValue, axis, values.tfjsArray.dtype)
}

suspend fun NDArrayTFJS.Companion.oneHot(indices: NumberNDArrayTFJS, depth: Int, offValue: Number, onValue: Number, axis: Int = -1, type: DataType): NDArrayTFJS {
    val dtype = type.resolveDTypeTFJS()
    return oneHot(indices, depth, offValue, onValue, axis, dtype)
}

internal suspend fun oneHot(indices: NumberNDArrayTFJS, depth: Int, offValue: Number, onValue: Number, axis: Int = -1, dtype: String): NDArrayTFJS {
    return tidyNDArray {
        val intIndices = indices.tfjsArray.cast("int32").dataInt()
        val actualIndices = NDArrayTFJS.int(indices.shapeArray) { it: Int ->
            val index = intIndices[it]
            if (index < 0) index + depth else index
        }

        val actualAxis = if (axis < 0) (indices.rank + 1) + axis else axis
        val oneHotLastAxis = oneHot(actualIndices.tfjsArray, depth, onValue, offValue, dtype)

        if (actualAxis != indices.rank) {
            val axes = (0..indices.rank).toTypedIntArray().apply {
                val tmp = this[lastIndex]
                this[lastIndex] = actualAxis
                this[actualAxis] = tmp
            }
            oneHotLastAxis.transpose(axes)
        } else {
            oneHotLastAxis
        }.toNDArray()

suspend fun NumberNDArrayTFJS.batchNorm(
    scale: NumberNDArrayTFJS,
    bias: NumberNDArrayTFJS,
    mean: NumberNDArrayTFJS,
    variance: NumberNDArrayTFJS,
    epsilon: Float
) : NumberNDArrayTFJS {
    val numChannels = this.shape.getOrNull(1) ?: 1

    require(scale.rank == 1 && scale.shape[0] == numChannels) { "\"scale\" must be a tensor of shape [$numChannels]" }
    require(bias.rank == 1 && bias.shape[0] == numChannels) { "\"bias\" must be a tensor of shape [$numChannels]" }
    require(mean.rank == 1 && mean.shape[0] == numChannels) { "\"mean\" must be a tensor of shape [$numChannels]" }
    require(variance.rank == 1 && variance.shape[0] == numChannels) { "\"variance\" must be a tensor of shape [$numChannels]" }
    return tidyNDArray {
        val indicesArray = shape.indices.toIntArray()
        val axes = intArrayOf(0) + if (rank > 2) indicesArray.sliceArray(2 until rank) else IntArray(0)

        val bcastScale = scale.unsqueeze(*axes).broadcastTo(shapeArray)
        val bcastBias = bias.unsqueeze(*axes).broadcastTo(shapeArray)
        val bcastMean = mean.unsqueeze(*axes).broadcastTo(shapeArray)
        val bcastVar = variance.unsqueeze(*axes).broadcastTo(shapeArray)
        NumberNDArrayTFJS(batchNorm(tfjsArray, bcastMean.tfjsArray, bcastVar.tfjsArray, bcastBias.tfjsArray, bcastScale.tfjsArray, epsilon))
    }
}
