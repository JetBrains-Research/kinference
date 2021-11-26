@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.tfjs.externals.core


internal external val add: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val broadcastTo: (x: NDArrayTFJS, shape: Array<Int>) -> NDArrayTFJS

internal external val cast: (x: NDArrayTFJS, dtype: String) -> NDArrayTFJS

internal external val reshape: (x: NDArrayTFJS, shape: Array<Int>) -> NDArrayTFJS

internal external val gather: (x: NDArrayTFJS, indices: NDArrayTFJS, axis: Int, batchDims: Int) -> NDArrayTFJS

internal external val moments: (x: NDArrayTFJS, axis: Array<Int>, keepDims: Boolean) -> MomentsOutput

internal external val sum: (x: NDArrayTFJS, axis: Array<Int>?, keepDims: Boolean) -> NDArrayTFJS

internal external val batchNorm: (x: NDArrayTFJS, mean: NDArrayTFJS, variance:  NDArrayTFJS, offset: NDArrayTFJS, scale: NDArrayTFJS, epsilon: Float) -> NDArrayTFJS

internal external val sqrt: (x: NDArrayTFJS) -> NDArrayTFJS

internal external val sub: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val div: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val mul: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val addN: (tensors: Array<NDArrayTFJS>) -> NDArrayTFJS

internal external val transpose: (x: NDArrayTFJS, perm: Array<Int>) -> NDArrayTFJS

internal external val unstack: (x: NDArrayTFJS, axis: Int) -> Array<NDArrayTFJS>

internal external val stack: (tensors: Array<NDArrayTFJS>, axis: Int) -> NDArrayTFJS

internal external val dot: (t1: NDArrayTFJS, t2: NDArrayTFJS) -> NDArrayTFJS

internal external val concat: (tensors: Array<NDArrayTFJS>, axis: Int) -> NDArrayTFJS

internal external val matMul: (a: NDArrayTFJS, b: NDArrayTFJS, transposeA: Boolean, transposeB: Boolean) -> NDArrayTFJS

internal external val softmax: (logits: NDArrayTFJS, dim: Int) -> NDArrayTFJS

internal external val erf: (x: NDArrayTFJS) -> NDArrayTFJS

internal external val min: (x: NDArrayTFJS, axis: Array<Int>?, keepDims: Boolean) -> NDArrayTFJS

internal external val max: (x: NDArrayTFJS, axis: Array<Int>?, keepDims: Boolean) -> NDArrayTFJS

internal external val round: (x: NDArrayTFJS) -> NDArrayTFJS

internal external val clipByValue: (x: NDArrayTFJS, clipValueMin: Number, clipValueMax: Number) -> NDArrayTFJS

internal external val neg: (x: NDArrayTFJS) -> NDArrayTFJS

internal external val minimum: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val maximum: (a: NDArrayTFJS, b: NDArrayTFJS) -> NDArrayTFJS

internal external val tanh: (x: NDArrayTFJS) -> NDArrayTFJS

internal external val slice: (x: NDArrayTFJS, begin: Array<Int>, size: Array<Int>?) -> NDArrayTFJS

internal external val reverse: (x: NDArrayTFJS, axis: Array<Int>?) -> NDArrayTFJS

internal external val stridedSlice: (x: NDArrayTFJS, begin: Array<Int>, end: Array<Int>, strides: Array<Int>?, beginMask: Int,
                                     endMask: Int, ellipsisMask: Int, newAxisMask: Int, shrinkAxisMask: Int) -> NDArrayTFJS

internal external val squeeze: (x: NDArrayTFJS, axis: Array<Int>?) -> NDArrayTFJS

internal external val argMax: (x: NDArrayTFJS, axis: Int) -> NDArrayTFJS
