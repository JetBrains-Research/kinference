@file:Suppress("UNCHECKED_CAST")

package io.kinference.ndarray.extensions

import io.kinference.ndarray.MomentsOutput
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.core.*
import io.kinference.ndarray.makeNDArray

fun ArrayTFJS.toNDArray() = makeNDArray(this, dtype)

fun Array<out NDArrayTFJS>.getArrays() = Array(this.size) { this[it].tfjsArray }
fun Array<out ArrayTFJS>.getNDArrays() = Array(this.size) { this[it].toNDArray() }

inline fun <T, V> T.innerCast(func: (V) -> V): T {
    this as? V ?: error { "Cannot perform cast" }
    return func(this) as T
}

inline fun <T : NumberNDArray> T.tfjs(func: (NumberNDArrayTFJS) -> NumberNDArrayTFJS): T = innerCast(func)

fun <T : NDArrayTFJS> T.dataInt() = tfjsArray.dataInt()
fun <T : NDArrayTFJS> T.dataFloat() = tfjsArray.dataFloat()
fun <T : NDArrayTFJS> T.dataBool() = tfjsArray.dataBool()

fun <T : NDArrayTFJS> T.broadcastTo(shape: Array<Int>) = tfjsArray.broadcastTo(shape).toNDArray() as T

fun <T : NDArrayTFJS> T.cast(dtype: String) = tfjsArray.cast(dtype).toNDArray()

fun <T : NDArrayTFJS> T.gather(indices: NDArrayTFJS, axis: Int = 0, batchDims: Int = 0) = tfjsArray.gather(indices.tfjsArray, axis, batchDims).toNDArray() as T

fun <T : NDArrayTFJS> Array<T>.concat(axis: Int = 0) = concat(getArrays(), axis).toNDArray() as T

fun <T : NDArrayTFJS> T.concat(vararg tensors: T, axis: Int = 0) = arrayOf(this, *tensors).concat(axis)

fun <T : NDArrayTFJS> T.transpose() = transpose(tfjsArray, null).toNDArray() as T

fun <T : NDArrayTFJS> T.unstack(axis: Int = 0) = unstack(tfjsArray, axis).getNDArrays() as Array<T>

fun <T : NDArrayTFJS> Array<T>.stack(axis: Int = 0) = stack(getArrays(), axis).toNDArray() as T

fun <T : NDArrayTFJS> Collection<T>.stack(axis: Int = 0) = this.toTypedArray().stack(0)

fun <T : NDArrayTFJS> T.flatten() = reshape(tfjsArray, arrayOf(this.linearSize)).toNDArray() as T

fun <T : NDArrayTFJS> T.stack(vararg tensors: NDArrayTFJS, axis: Int = 0) = arrayOf(tfjsArray, *tensors.getArrays()).stack(axis).toNDArray() as T

fun <T : NDArrayTFJS> T.slice(begin: Array<Int>, end: Array<Int>) = slice(tfjsArray, begin, end).toNDArray() as T

fun <T : NDArrayTFJS> T.slice(begin: Array<Int>) = slice(tfjsArray, begin, null).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse(axes: Array<Int>) = reverse(tfjsArray, axes).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse(axis: Int) = reverse(tfjsArray, arrayOf(axis)).toNDArray() as T

fun <T : NDArrayTFJS> T.reverse() = reverse(tfjsArray, null).toNDArray() as T

fun <T : NDArrayTFJS> T.slice(start: Array<Int>, end: Array<Int>, step: Array<Int>) = stridedSlice(tfjsArray, start, end, step, 0, 0, 0, 0, 0).toNDArray() as T

fun <T : NDArrayTFJS> T.squeeze(axes: Array<Int>? = null) = squeeze(tfjsArray, axes).toNDArray() as T

fun <T : NDArrayTFJS> T.equal(other: NDArrayTFJS) = BooleanNDArrayTFJS(equal(tfjsArray, other.tfjsArray))

fun <T : NDArrayTFJS> T.where(condition: NDArrayTFJS, other: NDArrayTFJS) = where(condition.tfjsArray, tfjsArray, other.tfjsArray).toNDArray()

fun <T : NDArrayTFJS> T.pad(paddings: Array<Array<Int>>, constantValue: Number) = pad(tfjsArray, paddings, constantValue).toNDArray() as T

fun <T : NDArrayTFJS> T.gatherNd(indices: NDArrayTFJS) = gatherND(tfjsArray, indices.tfjsArray).toNDArray()

fun <T : NDArrayTFJS> T.topk(k: Int, sorted: Boolean = false) = topk(tfjsArray, k, sorted).toNDArray()

fun NumberNDArrayTFJS.leakyRelu(alpha: Number) = NumberNDArrayTFJS(leakyRelu(tfjsArray, alpha))

fun NumberNDArrayTFJS.sum(axis: Int, keepDims: Boolean = false) = NumberNDArrayTFJS(sum(tfjsArray, arrayOf(axis), keepDims))

fun NumberNDArrayTFJS.sum(axes: Array<Int>, keepDims: Boolean = false) = NumberNDArrayTFJS(sum(tfjsArray, axes, keepDims))

fun Array<NumberNDArrayTFJS>.sum() = NumberNDArrayTFJS(addN(getArrays()))

fun NumberNDArrayTFJS.add(other: NumberNDArrayTFJS) = NumberNDArrayTFJS(add(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.add(tensors: Array<NumberNDArrayTFJS>) = NumberNDArrayTFJS(addN(arrayOf(tfjsArray, *tensors.getArrays())))

fun NumberNDArrayTFJS.add(vararg tensors: NumberNDArrayTFJS) = add(tensors as Array<NumberNDArrayTFJS>)

fun NumberNDArrayTFJS.dot(other: NumberNDArrayTFJS) = NumberNDArrayTFJS(dot(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.matMul(other: NumberNDArrayTFJS, transposeLeft: Boolean = false, transposeRight: Boolean = false) =
    NumberNDArrayTFJS(matMul(tfjsArray, other.tfjsArray, transposeLeft, transposeRight))

fun NumberNDArrayTFJS.softmax(axis: Int = -1) = NumberNDArrayTFJS(softmax(tfjsArray, axis))

fun NumberNDArrayTFJS.min(axes: Array<Int>, keepDims: Boolean = false) = NumberNDArrayTFJS(min(tfjsArray, axes, keepDims))

fun NumberNDArrayTFJS.min(keepDims: Boolean = false) = NumberNDArrayTFJS(min(tfjsArray, null, keepDims))

fun NumberNDArrayTFJS.max(keepDims: Boolean = false) = NumberNDArrayTFJS(max(tfjsArray, null, keepDims))

fun min(a: NumberNDArrayTFJS, b: NumberNDArrayTFJS) = NumberNDArrayTFJS(minimum(a.tfjsArray, b.tfjsArray))

fun max(a: NumberNDArrayTFJS, b: NumberNDArrayTFJS) = NumberNDArrayTFJS(maximum(a.tfjsArray, b.tfjsArray))

fun NumberNDArrayTFJS.round() = NumberNDArrayTFJS(round(tfjsArray))

fun NumberNDArrayTFJS.clip(minValue: Number, maxValue: Number) = NumberNDArrayTFJS(clipByValue(tfjsArray, minValue, maxValue))

operator fun NumberNDArrayTFJS.unaryMinus() = NumberNDArrayTFJS(neg(tfjsArray))

fun NumberNDArrayTFJS.sqrt() = NumberNDArrayTFJS(sqrt(tfjsArray))

fun sqrt(value: NumberNDArrayTFJS) = value.sqrt()

fun NumberNDArrayTFJS.tanh() = NumberNDArrayTFJS(tanh(tfjsArray))

fun NumberNDArrayTFJS.tanh(x: NumberNDArrayTFJS) = x.tanh()

fun NumberNDArrayTFJS.less(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(less(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.greater(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(greater(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.greaterEqual(other: NumberNDArrayTFJS) = BooleanNDArrayTFJS(greaterEqual(tfjsArray, other.tfjsArray))

fun NumberNDArrayTFJS.moments(axis: Int, keepDims: Boolean = false): MomentsOutput {
    val out = moments(this.tfjsArray, arrayOf(axis), keepDims)
    return MomentsOutput(out["mean"] as ArrayTFJS, out["variance"] as ArrayTFJS)
}

fun NumberNDArrayTFJS.moments(axes: Array<Int>, keepDims: Boolean = false): MomentsOutput {
    val out = moments(this.tfjsArray, axes, keepDims)
    return MomentsOutput(out["mean"] as ArrayTFJS, out["variance"] as ArrayTFJS)
}
