package io.kinference.tfjs.custom_externals.extensions

import io.kinference.tfjs.custom_externals.core.*
import org.khronos.webgl.*

fun tensor(values: FloatArray, shape: Array<Int>, dtype: String): TensorTFJS = tensor(values.unsafeCast<Float32Array>(), shape, dtype)

fun tensor(values: IntArray, shape: Array<Int>, dtype: String): TensorTFJS = tensor(values.unsafeCast<Int32Array>(), shape, dtype)

fun tensor(values: UByteArray, shape: Array<Int>, dtype: String): TensorTFJS = tensor(values.unsafeCast<Uint8Array>(), shape, dtype)

fun TensorTFJS.dataInt() = (dataSync() as Int32Array).unsafeCast<IntArray>()

fun TensorTFJS.dataFloat() = (dataSync() as Float32Array).unsafeCast<FloatArray>()

operator fun TensorTFJS.plus(other: TensorTFJS) = io.kinference.tfjs.custom_externals.core.add(this, other)

operator fun TensorTFJS.minus(other: TensorTFJS) = sub(this, other)

operator fun TensorTFJS.div(other: TensorTFJS) = div(this, other)

operator fun TensorTFJS.times(other: TensorTFJS) = mul(this, other)

fun TensorTFJS.broadcastTo(shape: Array<Int>) = broadcastTo(this, shape)

fun TensorTFJS.cast(dtype: String) = cast(this, dtype)

fun TensorTFJS.reshape(shape: Array<Int>) = reshape(this, shape)

fun TensorTFJS.gather(indices: TensorTFJS, axis: Int = 0, batchDims: Int = 0) = gather(this, indices, axis, batchDims)

fun TensorTFJS.moments(axis: Int, keepDims: Boolean = false) = moments(this, arrayOf(axis), keepDims)

fun TensorTFJS.moments(axes: Array<Int>, keepDims: Boolean = false) = moments(this, axes, keepDims)

fun TensorTFJS.sum(axis: Int, keepDims: Boolean = false) = sum(this, arrayOf(axis), keepDims)

fun TensorTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = sum(this, axes, keepDims)

fun TensorTFJS.sum(keepDims: Boolean = false) = sum(this, null, keepDims)

fun TensorTFJS.sqrt() = io.kinference.tfjs.custom_externals.core.sqrt(this)

fun Array<TensorTFJS>.sum() = addN(this)

fun TensorTFJS.add(vararg tensors: TensorTFJS) = addN(arrayOf(this, *tensors))

fun TensorTFJS.transpose(permutation: Array<Int>) = transpose(this, permutation)

fun TensorTFJS.unstack(axis: Int = 0) = unstack(this, axis)

fun Array<TensorTFJS>.stack(axis: Int = 0) = stack(this, axis)

fun TensorTFJS.stack(vararg tensors: TensorTFJS, axis: Int = 0) = stack(arrayOf(this, *tensors), axis)

fun TensorTFJS.dot(other: TensorTFJS) = dot(this, other)

fun Array<TensorTFJS>.concat(axis: Int = 0) = concat(this, axis)

fun TensorTFJS.concat(vararg tensors: TensorTFJS, axis: Int = 0) = concat(arrayOf(this, *tensors), axis)

fun TensorTFJS.matMul(other: TensorTFJS, transposeLeft: Boolean = false, transposeRight: Boolean = false) = matMul(this, other, transposeLeft, transposeRight)

fun TensorTFJS.softmax(axis: Int = -1) = softmax(this, axis)

fun TensorTFJS.erf() = erf(this)

fun TensorTFJS.flatten() = reshape(this, arrayOf(this.size))

fun TensorTFJS.isScalar() = shape.isEmpty()

fun TensorTFJS.computeBlockSize(fromDim: Int = 0, toDim: Int = this.shape.size): Int {
    return this.shape.sliceArray(fromDim until toDim).fold(1, Int::times)
}

fun TensorTFJS.indexAxis(axis: Int) = if (axis < 0) rank + axis else axis

fun TensorTFJS.min(axis: Int = 0, keepDims: Boolean = false) = min(this, arrayOf(axis), keepDims)

fun TensorTFJS.min(axes: Array<Int>, keepDims: Boolean = false) = min(this, axes, keepDims)

fun TensorTFJS.min(keepDims: Boolean = false) = min(this, null, keepDims)

fun TensorTFJS.max(axis: Int, keepDims: Boolean = false) = max(this, arrayOf(axis), keepDims)

fun TensorTFJS.max(axes: Array<Int>, keepDims: Boolean = false) = max(this, axes, keepDims)

fun TensorTFJS.max(keepDims: Boolean = false) = max(this, null, keepDims)

fun TensorTFJS.round() = round(this)

fun TensorTFJS.clip(minValue: Number, maxValue: Number) = clipByValue(this, minValue, maxValue)

operator fun TensorTFJS.unaryMinus() = neg(this)

fun min(a: TensorTFJS, b: TensorTFJS) = minimum(a, b)

fun max(a: TensorTFJS, b: TensorTFJS) = maximum(a, b)

fun sqrt(value: TensorTFJS) = value.sqrt()
