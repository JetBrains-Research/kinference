package io.kinference.tfjs.externals.extensions

import io.kinference.tfjs.externals.core.*
import org.khronos.webgl.*

fun tensor(values: FloatArray, shape: Array<Int>, dtype: String): NDArrayTFJS = tensor(values.unsafeCast<Float32Array>(), shape, dtype)

fun tensor(values: IntArray, shape: Array<Int>, dtype: String): NDArrayTFJS = tensor(values.unsafeCast<Int32Array>(), shape, dtype)

fun tensor(values: UByteArray, shape: Array<Int>, dtype: String): NDArrayTFJS = tensor(values.unsafeCast<Uint8Array>(), shape, dtype)

fun scalar(value: Boolean) = scalar(value, "bool")

fun scalar(value: Float) = scalar(value, "float32")

fun scalar(value: Int) = scalar(value, "int32")

fun NDArrayTFJS.dataInt() = dataSync().unsafeCast<Int32Array>().unsafeCast<IntArray>()

fun NDArrayTFJS.dataFloat() = dataSync().unsafeCast<Float32Array>().unsafeCast<FloatArray>()

fun NDArrayTFJS.dataBool() = dataSync().unsafeCast<Array<Boolean>>()

operator fun NDArrayTFJS.plus(other: NDArrayTFJS) = io.kinference.tfjs.externals.core.add(this, other)

operator fun NDArrayTFJS.minus(other: NDArrayTFJS) = sub(this, other)

operator fun NDArrayTFJS.div(other: NDArrayTFJS) = div(this, other)

operator fun NDArrayTFJS.times(other: NDArrayTFJS) = mul(this, other)

fun NDArrayTFJS.broadcastTo(shape: Array<Int>) = broadcastTo(this, shape)

fun NDArrayTFJS.cast(dtype: String) = cast(this, dtype)

fun NDArrayTFJS.reshape(shape: Array<Int>) = reshape(this, shape)

fun NDArrayTFJS.gather(indices: NDArrayTFJS, axis: Int = 0, batchDims: Int = 0) = gather(this, indices, axis, batchDims)

fun NDArrayTFJS.moments(axis: Int, keepDims: Boolean = false): MomentsOutput {
    val out = moments(this, arrayOf(axis), keepDims)
    return MomentsOutput(out["mean"] as NDArrayTFJS, out["variance"] as NDArrayTFJS)
}

fun NDArrayTFJS.moments(axes: Array<Int>, keepDims: Boolean = false): MomentsOutput {
    val out = moments(this, axes, keepDims)
    return MomentsOutput(out["mean"] as NDArrayTFJS, out["variance"] as NDArrayTFJS)
}

fun NDArrayTFJS.sum(axis: Int, keepDims: Boolean = false) = sum(this, arrayOf(axis), keepDims)

fun NDArrayTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = sum(this, axes, keepDims)

fun NDArrayTFJS.sum(keepDims: Boolean = false) = sum(this, null, keepDims)

fun NDArrayTFJS.sqrt() = io.kinference.tfjs.externals.core.sqrt(this)

fun Array<NDArrayTFJS>.sum() = addN(this)

fun NDArrayTFJS.add(tensors: Array<NDArrayTFJS>) = addN(arrayOf(this, *tensors))

fun NDArrayTFJS.add(vararg tensors: NDArrayTFJS) = addN(arrayOf(this, *tensors))

fun NDArrayTFJS.transpose() = transpose(this, null)

fun NDArrayTFJS.transpose(permutation: Array<Int>? = null) = transpose(this, permutation)

fun NDArrayTFJS.unstack(axis: Int = 0) = unstack(this, axis)

fun Array<NDArrayTFJS>.stack(axis: Int = 0) = stack(this, axis)

fun Collection<NDArrayTFJS>.stack(axis: Int = 0) = this.toTypedArray().stack(axis)

fun NDArrayTFJS.stack(vararg tensors: NDArrayTFJS, axis: Int = 0) = stack(arrayOf(this, *tensors), axis)

fun NDArrayTFJS.dot(other: NDArrayTFJS) = dot(this, other)

fun Array<NDArrayTFJS>.concat(axis: Int = 0) = concat(this, axis)

fun NDArrayTFJS.concat(vararg tensors: NDArrayTFJS, axis: Int = 0) = concat(arrayOf(this, *tensors), axis)

fun NDArrayTFJS.matMul(other: NDArrayTFJS, transposeLeft: Boolean = false, transposeRight: Boolean = false) = matMul(this, other, transposeLeft, transposeRight)

fun NDArrayTFJS.softmax(axis: Int = -1) = softmax(this, axis)

fun NDArrayTFJS.erf() = erf(this)

fun NDArrayTFJS.flatten() = reshape(this, arrayOf(this.size))

fun NDArrayTFJS.isScalar() = shape.isEmpty()

fun NDArrayTFJS.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

fun NDArrayTFJS.indexAxis(axis: Int) = if (axis < 0) rank + axis else axis

fun NDArrayTFJS.min(axis: Int = 0, keepDims: Boolean = false) = min(this, arrayOf(axis), keepDims)

fun NDArrayTFJS.min(axes: Array<Int>, keepDims: Boolean = false) = min(this, axes, keepDims)

fun NDArrayTFJS.min(keepDims: Boolean = false) = min(this, null, keepDims)

fun NDArrayTFJS.max(axis: Int, keepDims: Boolean = false) = max(this, arrayOf(axis), keepDims)

fun NDArrayTFJS.max(axes: Array<Int>, keepDims: Boolean = false) = max(this, axes, keepDims)

fun NDArrayTFJS.max(keepDims: Boolean = false) = max(this, null, keepDims)

fun NDArrayTFJS.round() = round(this)

fun NDArrayTFJS.clip(minValue: Number, maxValue: Number) = clipByValue(this, minValue, maxValue)

operator fun NDArrayTFJS.unaryMinus() = neg(this)

fun min(a: NDArrayTFJS, b: NDArrayTFJS) = minimum(a, b)

fun max(a: NDArrayTFJS, b: NDArrayTFJS) = maximum(a, b)

fun sqrt(value: NDArrayTFJS) = value.sqrt()

fun NDArrayTFJS.tanh() = io.kinference.tfjs.externals.core.tanh(this)

fun tanh(x: NDArrayTFJS) = x.tanh()

fun NDArrayTFJS.slice(begin: Array<Int>, end: Array<Int>) = slice(this, begin, end)

fun NDArrayTFJS.slice(begin: Array<Int>) = slice(this, begin, null)

fun NDArrayTFJS.reverse(axes: Array<Int>) = reverse(this, axes)

fun NDArrayTFJS.reverse(axis: Int) = reverse(this, arrayOf(axis))

fun NDArrayTFJS.reverse() = reverse(this, null)

fun NDArrayTFJS.slice(start: Array<Int>, end: Array<Int>, step: Array<Int>) = stridedSlice(this, start, end, step, 0, 0, 0, 0, 0)

fun NDArrayTFJS.squeeze(axes: Array<Int>? = null) = squeeze(this, axes)

fun NDArrayTFJS.argmax(axis: Int = 0) = argMax(this, axis)

fun NDArrayTFJS.tile(repeats: Array<Int>) = tile(this, repeats)

fun NDArrayTFJS.less(other: NDArrayTFJS) = less(this, other)

fun NDArrayTFJS.greater(other: NDArrayTFJS) = greater(this, other)

fun NDArrayTFJS.greaterEqual(other: NDArrayTFJS) = greaterEqual(this, other)

fun NDArrayTFJS.equal(other: NDArrayTFJS) = equal(this, other)

fun NDArrayTFJS.where(condition: NDArrayTFJS, other: NDArrayTFJS) = where(condition, this, other)

fun NDArrayTFJS.clone() = clone(this)

fun NDArrayTFJS.not(): NDArrayTFJS {
    require(this.dtype == "bool") { "Accepted only bool type" }
    return logicalNot(this)
}

fun NDArrayTFJS.pad(paddings: Array<Array<Int>>, constantValue: Number) = pad(this, paddings, constantValue)

fun NDArrayTFJS.pad(paddings: Array<Array<Int>>, constantValue: Boolean) = pad(this, paddings, constantValue)

fun NDArrayTFJS.gatherNd(indices: NDArrayTFJS) = gatherND(this, indices)

fun NDArrayTFJS.leakyRelu(alpha: Number) = leakyRelu(this, alpha)
