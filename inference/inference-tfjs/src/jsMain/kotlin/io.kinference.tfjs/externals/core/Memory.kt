@file:JsModule("@tensorflow/tfjs-core")
@file:JsNonModule

package io.kinference.tfjs.externals.core

external fun tidy(nameOrFn: () -> Array<NDArrayTFJS>, fn: (() -> Array<NDArrayTFJS>)?): Array<NDArrayTFJS>

external fun tidy(nameOrFn: String, fn: (() -> Array<NDArrayTFJS>)?): Array<NDArrayTFJS>


