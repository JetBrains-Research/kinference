@file:JsModule("@tensorflow/tfjs-core")

package io.kinference.custom_externals.core

external fun tidy(nameOrFn: () -> Array<TensorTFJS>, fn: (() -> Array<TensorTFJS>)?): Array<TensorTFJS>

external fun tidy(nameOrFn: String, fn: (() -> Array<TensorTFJS>)?): Array<TensorTFJS>

external fun memory(): MemoryInfo


