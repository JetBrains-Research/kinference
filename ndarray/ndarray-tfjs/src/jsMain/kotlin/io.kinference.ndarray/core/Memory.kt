@file:JsModule("@tensorflow/tfjs-core")
@file:JsNonModule

package io.kinference.ndarray.core

import io.kinference.ndarray.arrays.ArrayTFJS

external fun tidy(nameOrFn: () -> Array<ArrayTFJS>, fn: (() -> Array<ArrayTFJS>)?): Array<ArrayTFJS>

external fun tidy(nameOrFn: String, fn: (() -> Array<ArrayTFJS>)?): Array<ArrayTFJS>

external class Engine {
    fun startScope(name: String?)
    fun endScope(result: Array<ArrayTFJS>?)
}

external fun engine(): Engine

external fun memory(): MemoryInfo

external interface MemoryInfo {
    val numTensors: Int
    val numDataBuffers: Int
    val numBytes: Int
    val unreliable: Boolean?
    val reasons: Array<String>
}
