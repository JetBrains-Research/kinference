@file:JsModule("@tensorflow/tfjs-core")
@file:JsNonModule
package io.kinference.ndarray.core

import io.kinference.ndarray.MomentsOutputTFJS
import io.kinference.ndarray.arrays.ArrayTFJS
import kotlin.js.Promise

internal external val abs: (x: ArrayTFJS) -> ArrayTFJS

internal external val asin: (x: ArrayTFJS) -> ArrayTFJS

internal external val asinh: (x: ArrayTFJS) -> ArrayTFJS

internal external val atan: (x: ArrayTFJS) -> ArrayTFJS

internal external val atanh: (x: ArrayTFJS) -> ArrayTFJS

internal external val acos: (x: ArrayTFJS) -> ArrayTFJS

internal external val acosh: (x: ArrayTFJS) -> ArrayTFJS

internal external val add: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val broadcastTo: (x: ArrayTFJS, shape: Array<Int>) -> ArrayTFJS

internal external val cast: (x: ArrayTFJS, dtype: String) -> ArrayTFJS

internal external val reshape: (x: ArrayTFJS, shape: Array<Int>) -> ArrayTFJS

internal external val gather: (x: ArrayTFJS, indices: ArrayTFJS, axis: Int, batchDims: Int) -> ArrayTFJS

internal external val moments: (x: ArrayTFJS, axis: Array<Int>, keepDims: Boolean) -> MomentsOutputTFJS

internal external val sum: (x: ArrayTFJS, axis: Array<Int>?, keepDims: Boolean) -> ArrayTFJS

internal external val batchNorm: (x: ArrayTFJS, mean: ArrayTFJS, variance: ArrayTFJS, offset: ArrayTFJS, scale: ArrayTFJS, epsilon: Float) -> ArrayTFJS

internal external val sqrt: (x: ArrayTFJS) -> ArrayTFJS

internal external val sub: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val div: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val mul: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val addN: (tensors: Array<ArrayTFJS>) -> ArrayTFJS

internal external val transpose: (x: ArrayTFJS, perm: Array<Int>?) -> ArrayTFJS

internal external val unstack: (x: ArrayTFJS, axis: Int) -> Array<ArrayTFJS>

internal external val stack: (tensors: Array<ArrayTFJS>, axis: Int) -> ArrayTFJS

internal external val dot: (t1: ArrayTFJS, t2: ArrayTFJS) -> ArrayTFJS

internal external val concat: (tensors: Array<ArrayTFJS>, axis: Int) -> ArrayTFJS

internal external val split: (x: ArrayTFJS, numOrSizeSplits: dynamic, axis: Int) -> Array<ArrayTFJS>

internal external val matMul: (a: ArrayTFJS, b: ArrayTFJS, transposeA: Boolean, transposeB: Boolean) -> ArrayTFJS

internal external val softmax: (logits: ArrayTFJS, dim: Int) -> ArrayTFJS

internal external val logSoftmax: (logits: ArrayTFJS, axis: Int) -> ArrayTFJS

internal external val log: (x: ArrayTFJS) -> ArrayTFJS

internal external val erf: (x: ArrayTFJS) -> ArrayTFJS

internal external val min: (x: ArrayTFJS, axis: Array<Int>?, keepDims: Boolean?) -> ArrayTFJS

internal external val max: (x: ArrayTFJS, axis: Array<Int>?, keepDims: Boolean?) -> ArrayTFJS

internal external val round: (x: ArrayTFJS) -> ArrayTFJS

internal external val clipByValue: (x: ArrayTFJS, clipValueMin: Number, clipValueMax: Number) -> ArrayTFJS

internal external val neg: (x: ArrayTFJS) -> ArrayTFJS

internal external val minimum: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val maximum: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val tanh: (x: ArrayTFJS) -> ArrayTFJS

internal external val slice: (x: ArrayTFJS, begin: Array<Int>, size: Array<Int>?) -> ArrayTFJS

internal external val reverse: (x: ArrayTFJS, axis: Array<Int>?) -> ArrayTFJS

internal external val stridedSlice: (
    x: ArrayTFJS, begin: Array<Int>, end: Array<Int>, strides: Array<Int>?, beginMask: Int,
    endMask: Int, ellipsisMask: Int, newAxisMask: Int, shrinkAxisMask: Int
) -> ArrayTFJS

internal external val squeeze: (x: ArrayTFJS, axis: Array<Int>?) -> ArrayTFJS

internal external val argMax: (x: ArrayTFJS, axis: Int) -> ArrayTFJS

internal external val argMin: (x: ArrayTFJS, axis: Int) -> ArrayTFJS

internal external val tile: (x: ArrayTFJS, reps: Array<Int>) -> ArrayTFJS

internal external val less: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val greater: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val greaterEqual: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val equal: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val notEqual: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val where: (condition: ArrayTFJS, a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val whereAsync: (condition: ArrayTFJS) -> Promise<ArrayTFJS>

internal external val clone: (x: ArrayTFJS) -> ArrayTFJS

internal external val logicalNot: (x: ArrayTFJS) -> ArrayTFJS

internal external val logicalOr: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val logicalAnd: (a: ArrayTFJS, b: ArrayTFJS) -> ArrayTFJS

internal external val pad: (x: ArrayTFJS, paddings: Array<Array<Int>>, constantValue: dynamic) -> ArrayTFJS

internal external val mirrorPad: (x: ArrayTFJS, paddings: Array<Array<Int>>, mode: String) -> ArrayTFJS

internal external val gatherND: (x: ArrayTFJS, indices: ArrayTFJS) -> ArrayTFJS

internal external val leakyRelu: (x: ArrayTFJS, alpha: Number) -> ArrayTFJS

internal external val relu: (x: ArrayTFJS) -> ArrayTFJS

internal external val cumsum: (x: ArrayTFJS, axis: Int, exclusive: Boolean, reverse: Boolean) -> ArrayTFJS

internal external val topk: (x: ArrayTFJS, k: Int, sorted: Boolean) -> Pair<ArrayTFJS, ArrayTFJS>

internal external val sigmoid: (x: ArrayTFJS) -> ArrayTFJS

internal external val tensorScatterUpdate: (tensor: ArrayTFJS, indices: ArrayTFJS, updates: ArrayTFJS) -> ArrayTFJS

internal external val ceil: (x: ArrayTFJS) -> ArrayTFJS

internal external val exp: (x: ArrayTFJS) -> ArrayTFJS

internal external val expm1: (x: ArrayTFJS) -> ArrayTFJS

internal external val elu: (x: ArrayTFJS) -> ArrayTFJS

internal external val prelu: (x: ArrayTFJS, alpha: ArrayTFJS) -> ArrayTFJS

internal external val cos: (x: ArrayTFJS) -> ArrayTFJS

internal external val cosh: (x: ArrayTFJS) -> ArrayTFJS

internal external val prod: (x: ArrayTFJS, axis: Array<Int>, keepDims: Boolean) -> ArrayTFJS

internal external val linalg: Linalg

internal external val floor: (x: ArrayTFJS) -> ArrayTFJS

internal external val isInf: (x: ArrayTFJS) -> ArrayTFJS

internal external val isNaN: (x: ArrayTFJS) -> ArrayTFJS
