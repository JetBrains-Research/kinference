@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.custom_externals.core


internal external val add: (a: TensorTFJS, b: TensorTFJS) -> TensorTFJS

internal external val broadcastTo: (x: TensorTFJS, shape: Array<Int>) -> TensorTFJS

internal external val cast: (x: TensorTFJS, dtype: String) -> TensorTFJS

internal external val reshape: (x: TensorTFJS, shape: Array<Int>) -> TensorTFJS

internal external val gather: (x: TensorTFJS, indices: TensorTFJS, axis: Int, batchDims: Int) -> TensorTFJS

internal external val moments: (x: TensorTFJS, axis: Array<Int>, keepDims: Boolean) -> MomentsOutput

internal external val sum: (x: TensorTFJS, axis: Array<Int>, keepDims: Boolean) -> TensorTFJS

internal external val batchNorm: (x: TensorTFJS, mean: TensorTFJS, variance:  TensorTFJS, offset: TensorTFJS, scale: TensorTFJS, epsilon: Float) -> TensorTFJS

internal external val sqrt: (x: TensorTFJS) -> TensorTFJS

internal external val sub: (a: TensorTFJS, b: TensorTFJS) -> TensorTFJS

internal external val div: (a: TensorTFJS, b: TensorTFJS) -> TensorTFJS

internal external val mul: (a: TensorTFJS, b: TensorTFJS) -> TensorTFJS

internal external val addN: (tensors: Array<TensorTFJS>) -> TensorTFJS

internal external val transpose: (x: TensorTFJS, perm: Array<Int>) -> TensorTFJS

internal external val unstack: (x: TensorTFJS, axis: Int) -> Array<TensorTFJS>

internal external val stack: (tensors: Array<TensorTFJS>, axis: Int) -> TensorTFJS

internal external val dot: (t1: TensorTFJS, t2: TensorTFJS) -> TensorTFJS

internal external val concat: (tensors: Array<TensorTFJS>, axis: Int) -> TensorTFJS

internal external val matMul: (a: TensorTFJS, b: TensorTFJS, transposeA: Boolean, transposeB: Boolean) -> TensorTFJS

internal external val softmax: (logits: TensorTFJS, dim: Int) -> TensorTFJS

internal external val erf: (x: TensorTFJS) -> TensorTFJS
