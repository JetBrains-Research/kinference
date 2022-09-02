@file:JsModule("@tensorflow/tfjs-core")
@file:JsNonModule

package io.kinference.ndarray.core

import io.kinference.ndarray.arrays.ArrayTFJS

external fun tidy(nameOrFn: () -> Array<ArrayTFJS>, fn: (() -> Array<ArrayTFJS>)?): Array<ArrayTFJS>

external fun tidy(nameOrFn: String, fn: (() -> Array<ArrayTFJS>)?): Array<ArrayTFJS>
