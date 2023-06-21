package io.kinference.ndarray.extensions

import io.kinference.ndarray.MomentsOutput
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

fun tensor(values: FloatArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Float32Array>(), shape, dtype)

fun tensor(values: IntArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Int32Array>(), shape, dtype)

fun tensor(values: UByteArray, shape: Array<Int>, dtype: String): ArrayTFJS = tensor(values.unsafeCast<Uint8Array>(), shape, dtype)

fun tensor(values: Array<Boolean>, shape: Array<Int>) = tensor(values, shape, "bool")

fun scalar(value: Boolean) = scalar(value, "bool")

fun scalar(value: Float) = scalar(value, "float32")

fun scalar(value: Int) = scalar(value, "int32")

fun ArrayTFJS.dataInt() = dataSync().unsafeCast<Int32Array>().unsafeCast<IntArray>()

fun ArrayTFJS.dataFloat() = dataSync().unsafeCast<Float32Array>().unsafeCast<FloatArray>()

fun ArrayTFJS.dataBool() = dataSync().unsafeCast<Array<Boolean>>()

operator fun ArrayTFJS.plus(other: ArrayTFJS) = io.kinference.ndarray.core.add(this, other)

operator fun ArrayTFJS.minus(other: ArrayTFJS) = sub(this, other)

operator fun ArrayTFJS.div(other: ArrayTFJS) = div(this, other)

operator fun ArrayTFJS.times(other: ArrayTFJS) = mul(this, other)

fun ArrayTFJS.broadcastTo(shape: Array<Int>) = broadcastTo(this, shape)

fun ArrayTFJS.cast(dtype: String) = cast(this, dtype)

fun ArrayTFJS.reshape(shape: Array<Int>) = reshape(this, shape)
fun ArrayTFJS.reshape(shape: IntArray) = reshape(this, shape.toTypedArray())

fun ArrayTFJS.gather(indices: ArrayTFJS, axis: Int = 0, batchDims: Int = 0) = gather(this, indices, axis, batchDims)

fun ArrayTFJS.moments(axis: Int, keepDims: Boolean = false) = moments(this, arrayOf(axis), keepDims)

fun ArrayTFJS.moments(axes: Array<Int>, keepDims: Boolean = false) = moments(this, axes, keepDims)

fun ArrayTFJS.sum(axis: Int, keepDims: Boolean = false) = sum(this, arrayOf(axis), keepDims)

fun ArrayTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = sum(this, axes, keepDims)

fun ArrayTFJS.sum(keepDims: Boolean = false) = sum(this, null, keepDims)

fun Array<ArrayTFJS>.sum() = addN(this)

fun ArrayTFJS.add(tensors: Array<ArrayTFJS>) = addN(arrayOf(this, *tensors))

fun ArrayTFJS.add(vararg tensors: ArrayTFJS) = addN(arrayOf(this, *tensors))

fun ArrayTFJS.transpose() = transpose(this, null)

fun ArrayTFJS.transpose(permutation: Array<Int>? = null) = transpose(this, permutation)

fun ArrayTFJS.unstack(axis: Int = 0) = unstack(this, axis)

fun Array<ArrayTFJS>.stack(axis: Int = 0) = stack(this, axis)

fun Collection<ArrayTFJS>.stack(axis: Int = 0) = this.toTypedArray().stack(axis)

fun ArrayTFJS.stack(vararg tensors: ArrayTFJS, axis: Int = 0) = stack(arrayOf(this, *tensors), axis)

fun ArrayTFJS.dot(other: ArrayTFJS) = dot(this, other)

fun Array<ArrayTFJS>.concat(axis: Int = 0) = concat(this, axis)

fun ArrayTFJS.concat(vararg tensors: ArrayTFJS, axis: Int = 0) = concat(arrayOf(this, *tensors), axis)

fun ArrayTFJS.split(split: Array<Int>, axis: Int) = split(this, split, axis)

fun ArrayTFJS.split(splitSize: Int, axis: Int) = split(this, splitSize, axis)

fun ArrayTFJS.matMul(other: ArrayTFJS, transposeLeft: Boolean = false, transposeRight: Boolean = false) = matMul(this, other, transposeLeft, transposeRight)

fun ArrayTFJS.softmax(axis: Int = -1) = softmax(this, axis)

fun ArrayTFJS.logSoftmax(axis: Int = -1) = io.kinference.ndarray.core.logSoftmax(this , axis)

fun ArrayTFJS.log() = log(this)

fun ArrayTFJS.erf() = erf(this)

fun ArrayTFJS.flatten() = reshape(this, arrayOf(this.size))

fun ArrayTFJS.isScalar() = shape.isEmpty()

fun ArrayTFJS.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

fun ArrayTFJS.indexAxis(axis: Int) = if (axis < 0) rank + axis else axis

fun ArrayTFJS.min(axis: Int = 0, keepDims: Boolean = false) = min(this, arrayOf(axis), keepDims)

fun ArrayTFJS.min(axes: Array<Int>, keepDims: Boolean = false) = min(this, axes, keepDims)

fun ArrayTFJS.min(keepDims: Boolean = false) = min(this, null, keepDims)

fun ArrayTFJS.min() = min(this, null, null)

fun ArrayTFJS.max(axis: Int, keepDims: Boolean = false) = max(this, arrayOf(axis), keepDims)

fun ArrayTFJS.max(axes: Array<Int>, keepDims: Boolean = false) = max(this, axes, keepDims)

fun ArrayTFJS.max(keepDims: Boolean = false) = max(this, null, keepDims)

fun ArrayTFJS.max() = max(this, null, null)

fun ArrayTFJS.round() = round(this)

fun ArrayTFJS.clip(minValue: Number, maxValue: Number) = clipByValue(this, minValue, maxValue)

operator fun ArrayTFJS.unaryMinus() = neg(this)

fun min(a: ArrayTFJS, b: ArrayTFJS) = minimum(a, b)

fun max(a: ArrayTFJS, b: ArrayTFJS) = maximum(a, b)

fun ArrayTFJS.sqrt() = sqrt(this)

fun sqrt(value: ArrayTFJS) = value.sqrt()

fun ArrayTFJS.tanh() = tanh(this)

fun tanh(x: ArrayTFJS) = x.tanh()

fun ArrayTFJS.slice(begin: Array<Int>, end: Array<Int>) = slice(this, begin, end)

fun ArrayTFJS.slice(begin: Array<Int>) = slice(this, begin, null)

fun ArrayTFJS.reverse(axes: Array<Int>) = reverse(this, axes)

fun ArrayTFJS.reverse(axis: Int) = reverse(this, arrayOf(axis))

fun ArrayTFJS.reverse() = reverse(this, null)

fun ArrayTFJS.slice(start: Array<Int>, end: Array<Int>, step: Array<Int>) = stridedSlice(this, start, end, step, 0, 0, 0, 0, 0)

fun ArrayTFJS.squeeze(axes: Array<Int>? = null) = squeeze(this, axes)

fun ArrayTFJS.argmax(axis: Int = 0) = argMax(this, axis)

fun ArrayTFJS.argmin(axis: Int = 0) = argMin(this, axis)

fun ArrayTFJS.tile(repeats: Array<Int>) = tile(this, repeats)

fun ArrayTFJS.less(other: ArrayTFJS) = less(this, other)

fun ArrayTFJS.greater(other: ArrayTFJS) = greater(this, other)

fun ArrayTFJS.greaterEqual(other: ArrayTFJS) = greaterEqual(this, other)

fun ArrayTFJS.equal(other: ArrayTFJS) = equal(this, other)

fun ArrayTFJS.notEqual(other: ArrayTFJS) = notEqual(this, other)

fun ArrayTFJS.where(condition: ArrayTFJS, other: ArrayTFJS) = where(condition, this, other)

fun ArrayTFJS.clone() = clone(this)

fun ArrayTFJS.not(): ArrayTFJS {
    require(this.dtype == "bool") { "Only bool type is accepted" }
    return logicalNot(this)
}

fun ArrayTFJS.or(other: ArrayTFJS): ArrayTFJS {
    require(this.dtype == "bool" && other.dtype == "bool") { "Only boolean arrays are accepted" }
    return logicalOr(this, other)
}

fun ArrayTFJS.and(other: ArrayTFJS): ArrayTFJS {
    require(this.dtype == "bool" && other.dtype == "bool") { "Only boolean arrays are accepted" }
    return logicalAnd(this, other)
}

fun ArrayTFJS.pad(paddings: Array<Array<Int>>, constantValue: Any) = pad(this, paddings, constantValue)

internal fun ArrayTFJS.mirrorPad(paddings: Array<Array<Int>>, mode: String) = mirrorPad(this, paddings, mode)

fun ArrayTFJS.reflectPad(paddings: Array<Array<Int>>) = mirrorPad(this, paddings, "reflect")

fun ArrayTFJS.symmetricPad(paddings: Array<Array<Int>>) = mirrorPad(this, paddings, "symmetric")

fun ArrayTFJS.gatherNd(indices: ArrayTFJS) = gatherND(this, indices)

fun ArrayTFJS.leakyRelu(alpha: Number) = leakyRelu(this, alpha)

fun ArrayTFJS.relu() = relu(this)

fun ArrayTFJS.cumsum(axis: Int = 0, exclusive: Boolean = false, reverse: Boolean = false) =
    cumsum(this, axis, exclusive, reverse)

fun ArrayTFJS.topk(k: Int, sorted: Boolean = false) = topk(this, k, sorted)

fun ArrayTFJS.abs() = abs(this)

fun ArrayTFJS.acos() = acos(this)

fun ArrayTFJS.acosh() = acosh(this)

fun ArrayTFJS.asin() = asin(this)

fun ArrayTFJS.asinh() = asinh(this)

fun ArrayTFJS.atan() = atan(this)

fun ArrayTFJS.atanh() = atanh(this)

fun ArrayTFJS.tensorScatterUpdate(indices: ArrayTFJS, updates: ArrayTFJS) = tensorScatterUpdate(this, indices, updates)

fun ArrayTFJS.ceil() = ceil(this)
