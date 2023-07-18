package io.kinference.ndarray.extensions

import io.kinference.ndarray.QrDecompositionResultTFJS
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.core.*
import io.kinference.ndarray.core.broadcastTo
import io.kinference.ndarray.core.cast
import io.kinference.ndarray.core.clone
import io.kinference.ndarray.core.concat
import io.kinference.ndarray.core.cumsum
import io.kinference.ndarray.core.div
import io.kinference.ndarray.core.dot
import io.kinference.ndarray.core.equal
import io.kinference.ndarray.core.erf
import io.kinference.ndarray.core.gather
import io.kinference.ndarray.core.greater
import io.kinference.ndarray.core.greaterEqual
import io.kinference.ndarray.core.leakyRelu
import io.kinference.ndarray.core.less
import io.kinference.ndarray.core.matMul
import io.kinference.ndarray.core.max
import io.kinference.ndarray.core.min
import io.kinference.ndarray.core.moments
import io.kinference.ndarray.core.pad
import io.kinference.ndarray.core.reshape
import io.kinference.ndarray.core.reverse
import io.kinference.ndarray.core.round
import io.kinference.ndarray.core.slice
import io.kinference.ndarray.core.softmax
import io.kinference.ndarray.core.split
import io.kinference.ndarray.core.sqrt
import io.kinference.ndarray.core.squeeze
import io.kinference.ndarray.core.stack
import io.kinference.ndarray.core.sum
import io.kinference.ndarray.core.tanh
import io.kinference.ndarray.core.tile
import io.kinference.ndarray.core.topk
import io.kinference.ndarray.core.transpose
import io.kinference.ndarray.core.unstack
import io.kinference.ndarray.core.where
import org.khronos.webgl.*

internal fun tensor(values: FloatArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Float32Array>(), shape, dtype)

internal fun tensor(values: IntArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Int32Array>(), shape, dtype)

internal fun tensor(values: UByteArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Uint8Array>(), shape, dtype)

internal fun tensor(values: Array<Boolean>, shape: Array<Int>) = tensor(values, shape, "bool")

internal fun tensor(values: Array<String>, shape: Array<Int>) = tensor(values, shape, "string")

internal fun scalar(value: Boolean) = scalar(value, "bool")

internal fun scalar(value: Float) = scalar(value, "float32")

internal fun scalar(value: Int) = scalar(value, "int32")

internal fun scalar(value: String) = scalar(value, "string")

internal fun fill(shape: Array<Int>, value: String) = fill(shape, value, "string")

internal fun ArrayTFJS.dataInt() = dataSync().unsafeCast<Int32Array>().unsafeCast<IntArray>()

internal fun ArrayTFJS.dataFloat() = dataSync().unsafeCast<Float32Array>().unsafeCast<FloatArray>()

internal fun ArrayTFJS.dataBool() = dataSync().unsafeCast<Array<Boolean>>()

internal fun ArrayTFJS.dataString() = dataSync().unsafeCast<Array<String>>()

internal operator fun ArrayTFJS.plus(other: ArrayTFJS) = io.kinference.ndarray.core.add(this, other)

internal operator fun ArrayTFJS.minus(other: ArrayTFJS) = sub(this, other)

internal operator fun ArrayTFJS.div(other: ArrayTFJS) = div(this, other)

internal operator fun ArrayTFJS.times(other: ArrayTFJS) = mul(this, other)

internal fun ArrayTFJS.broadcastTo(shape: Array<Int>) = broadcastTo(this, shape)

internal fun ArrayTFJS.cast(dtype: String) = cast(this, dtype)

internal fun ArrayTFJS.reshape(shape: Array<Int>) = reshape(this, shape)
internal fun ArrayTFJS.reshape(shape: IntArray) = reshape(this, shape.toTypedArray())

internal fun ArrayTFJS.gather(indices: ArrayTFJS, axis: Int = 0, batchDims: Int = 0) = gather(this, indices, axis, batchDims)

internal fun ArrayTFJS.moments(axis: Int, keepDims: Boolean = false) = moments(this, arrayOf(axis), keepDims)

internal fun ArrayTFJS.moments(axes: Array<Int>, keepDims: Boolean = false) = moments(this, axes, keepDims)

internal fun ArrayTFJS.sum(axis: Int, keepDims: Boolean = false) = sum(this, arrayOf(axis), keepDims)

internal fun ArrayTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = sum(this, axes, keepDims)

internal fun ArrayTFJS.sum(keepDims: Boolean = false) = sum(this, null, keepDims)

internal fun Array<ArrayTFJS>.sum() = addN(this)

internal fun ArrayTFJS.add(tensors: Array<ArrayTFJS>) = addN(arrayOf(this, *tensors))

internal fun ArrayTFJS.add(vararg tensors: ArrayTFJS) = addN(arrayOf(this, *tensors))

internal fun ArrayTFJS.transpose() = transpose(this, null)

internal fun ArrayTFJS.transpose(permutation: Array<Int>? = null) = transpose(this, permutation)

internal fun ArrayTFJS.unstack(axis: Int = 0) = unstack(this, axis)

internal fun Array<ArrayTFJS>.stack(axis: Int = 0) = stack(this, axis)

internal fun Collection<ArrayTFJS>.stack(axis: Int = 0) = this.toTypedArray().stack(axis)

internal fun ArrayTFJS.stack(vararg tensors: ArrayTFJS, axis: Int = 0) = stack(arrayOf(this, *tensors), axis)

internal fun ArrayTFJS.dot(other: ArrayTFJS) = dot(this, other)

internal fun Array<ArrayTFJS>.concat(axis: Int = 0) = concat(this, axis)

internal fun ArrayTFJS.concat(vararg tensors: ArrayTFJS, axis: Int = 0) = concat(arrayOf(this, *tensors), axis)

internal fun ArrayTFJS.split(split: Array<Int>, axis: Int) = split(this, split, axis)

internal fun ArrayTFJS.split(splitSize: Int, axis: Int) = split(this, splitSize, axis)

internal fun ArrayTFJS.matMul(other: ArrayTFJS, transposeLeft: Boolean = false, transposeRight: Boolean = false) = matMul(this, other, transposeLeft, transposeRight)

internal fun ArrayTFJS.softmax(axis: Int = -1) = softmax(this, axis)

internal fun ArrayTFJS.logSoftmax(axis: Int = -1) = logSoftmax(this , axis)

internal fun ArrayTFJS.log() = log(this)

internal fun ArrayTFJS.erf() = erf(this)

internal fun ArrayTFJS.flatten() = reshape(this, arrayOf(this.size))

internal fun ArrayTFJS.isScalar() = shape.isEmpty()

internal fun ArrayTFJS.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

internal fun ArrayTFJS.indexAxis(axis: Int) = if (axis < 0) rank + axis else axis

internal fun ArrayTFJS.min(axis: Int = 0, keepDims: Boolean = false) = min(this, arrayOf(axis), keepDims)

internal fun ArrayTFJS.min(axes: Array<Int>, keepDims: Boolean = false) = min(this, axes, keepDims)

internal fun ArrayTFJS.min(keepDims: Boolean = false) = min(this, null, keepDims)

internal fun ArrayTFJS.min() = min(this, null, null)

internal fun ArrayTFJS.max(axis: Int, keepDims: Boolean = false) = max(this, arrayOf(axis), keepDims)

internal fun ArrayTFJS.max(axes: Array<Int>, keepDims: Boolean = false) = max(this, axes, keepDims)

internal fun ArrayTFJS.max(keepDims: Boolean = false) = max(this, null, keepDims)

internal fun ArrayTFJS.max() = max(this, null, null)

internal fun ArrayTFJS.round() = round(this)

internal fun ArrayTFJS.clip(minValue: Number, maxValue: Number) = clipByValue(this, minValue, maxValue)

internal operator fun ArrayTFJS.unaryMinus() = neg(this)

internal fun min(a: ArrayTFJS, b: ArrayTFJS) = minimum(a, b)

internal fun max(a: ArrayTFJS, b: ArrayTFJS) = maximum(a, b)

internal fun ArrayTFJS.sqrt() = sqrt(this)

internal fun sqrt(value: ArrayTFJS) = value.sqrt()

internal fun ArrayTFJS.tanh() = tanh(this)

internal fun tanh(x: ArrayTFJS) = x.tanh()

internal fun ArrayTFJS.slice(begin: Array<Int>, end: Array<Int>) = slice(this, begin, end)

internal fun ArrayTFJS.slice(begin: Array<Int>) = slice(this, begin, null)

internal fun ArrayTFJS.reverse(axes: Array<Int>) = reverse(this, axes)

internal fun ArrayTFJS.reverse(axis: Int) = reverse(this, arrayOf(axis))

internal fun ArrayTFJS.reverse() = reverse(this, null)

internal fun ArrayTFJS.slice(start: Array<Int>, end: Array<Int>, step: Array<Int>) = stridedSlice(this, start, end, step, 0, 0, 0, 0, 0)

internal fun ArrayTFJS.squeeze(axes: Array<Int>? = null) = squeeze(this, axes)

internal fun ArrayTFJS.argmax(axis: Int = 0) = argMax(this, axis)

internal fun ArrayTFJS.argmin(axis: Int = 0) = argMin(this, axis)

internal fun ArrayTFJS.tile(repeats: Array<Int>) = tile(this, repeats)

internal fun ArrayTFJS.less(other: ArrayTFJS) = less(this, other)

internal fun ArrayTFJS.greater(other: ArrayTFJS) = greater(this, other)

internal fun ArrayTFJS.greaterEqual(other: ArrayTFJS) = greaterEqual(this, other)

internal fun ArrayTFJS.equal(other: ArrayTFJS) = equal(this, other)

internal fun ArrayTFJS.notEqual(other: ArrayTFJS) = notEqual(this, other)

internal fun ArrayTFJS.where(condition: ArrayTFJS, other: ArrayTFJS) = where(condition, this, other)

internal fun ArrayTFJS.clone() = clone(this)

internal fun ArrayTFJS.not(): ArrayTFJS {
    require(this.dtype == "bool") { "Only bool type is accepted" }
    return logicalNot(this)
}

internal fun ArrayTFJS.or(other: ArrayTFJS): ArrayTFJS {
    require(this.dtype == "bool" && other.dtype == "bool") { "Only boolean arrays are accepted" }
    return logicalOr(this, other)
}

internal fun ArrayTFJS.and(other: ArrayTFJS): ArrayTFJS {
    require(this.dtype == "bool" && other.dtype == "bool") { "Only boolean arrays are accepted" }
    return logicalAnd(this, other)
}

internal fun ArrayTFJS.xor(other: ArrayTFJS): ArrayTFJS {
    require(this.dtype == "bool" && other.dtype == "bool") { "Only boolean arrays are accepted" }
    return logicalXor(this, other)
}

internal fun ArrayTFJS.pad(paddings: Array<Array<Int>>, constantValue: Any) = pad(this, paddings, constantValue)

internal fun ArrayTFJS.mirrorPad(paddings: Array<Array<Int>>, mode: String) = mirrorPad(this, paddings, mode)

internal fun ArrayTFJS.reflectPad(paddings: Array<Array<Int>>) = mirrorPad(this, paddings, "reflect")

internal fun ArrayTFJS.symmetricPad(paddings: Array<Array<Int>>) = mirrorPad(this, paddings, "symmetric")

internal fun ArrayTFJS.gatherNd(indices: ArrayTFJS) = gatherND(this, indices)

internal fun ArrayTFJS.leakyRelu(alpha: Number) = leakyRelu(this, alpha)

internal fun ArrayTFJS.relu() = relu(this)

internal fun ArrayTFJS.cumsum(axis: Int = 0, exclusive: Boolean = false, reverse: Boolean = false) =
    cumsum(this, axis, exclusive, reverse)

internal fun ArrayTFJS.topk(k: Int, sorted: Boolean = false) = topk(this, k, sorted)

internal fun ArrayTFJS.abs() = abs(this)

internal fun ArrayTFJS.acos() = acos(this)

internal fun ArrayTFJS.acosh() = acosh(this)

internal fun ArrayTFJS.asin() = asin(this)

internal fun ArrayTFJS.asinh() = asinh(this)

internal fun ArrayTFJS.sinh() = sinh(this)

internal fun ArrayTFJS.atan() = atan(this)

internal fun ArrayTFJS.atanh() = atanh(this)

internal fun ArrayTFJS.tan() = tan(this)

internal fun ArrayTFJS.tensorScatterUpdate(indices: ArrayTFJS, updates: ArrayTFJS) = tensorScatterUpdate(this, indices, updates)

internal fun ArrayTFJS.ceil() = ceil(this)

internal fun ArrayTFJS.exp() = exp(this)

internal fun ArrayTFJS.expm1() = expm1(this)

internal fun ArrayTFJS.elu() = elu(this)

internal fun ArrayTFJS.prelu(alpha: ArrayTFJS) = prelu(this, alpha)

internal fun ArrayTFJS.cos() = cos(this)

internal fun ArrayTFJS.cosh() = cosh(this)

internal fun ArrayTFJS.qrDecomposition(fullMatrices: Boolean = false): QrDecompositionResultTFJS {
    val result = linalg.qr(this, fullMatrices)
    return QrDecompositionResultTFJS(result[0], result[1])
}

internal fun ArrayTFJS.prod(axis: Int, keepDims: Boolean = false) = prod(this, arrayOf(axis), keepDims)

internal fun ArrayTFJS.prod(axes: Array<Int>, keepDims: Boolean = false) = prod(this, axes, keepDims)

internal fun ArrayTFJS.floor() = floor(this)

internal fun ArrayTFJS.isInf() = isInf(this)

internal fun ArrayTFJS.isNaN() = isNaN(this)

internal fun ArrayTFJS.bandPart(numLower: Int = 0, numUpper: Int = 0) = linalg.bandPart(this, numLower, numUpper)

internal fun ArrayTFJS.sin() = sin(this)
