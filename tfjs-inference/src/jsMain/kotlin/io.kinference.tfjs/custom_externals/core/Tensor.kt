@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.tfjs.custom_externals.core

import org.khronos.webgl.*
import kotlin.js.Promise


@JsName("Tensor")
open external class TensorTFJS {
    val shape: Array<Int>
    val size: Int
    val dtype: String /* "float32" | "int32" | "bool" | "complex64" | "string" */
    val rank: Int
    internal fun data(): Promise<Any>
    internal fun dataSync(): dynamic
    fun print(verbose: Boolean = definedExternally)
    fun dispose()
}


external fun tensor(values: Float32Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Int32Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Uint8Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Array<Int>, shape: Array<Int>, dtype: String): TensorTFJS
external fun tensor(values: Array<Float>, shape: Array<Int>, dtype: String): TensorTFJS
external fun tensor(values: Array<Double>, shape: Array<Int>, dtype: String): TensorTFJS
external fun tensor(values: Array<Byte>, shape: Array<Int>, dtype: String): TensorTFJS
external fun tensor(values: Array<UByte>, shape: Array<Int>, dtype: String): TensorTFJS

external fun range(start: Number, stop: Number, step: Number?, dtype: String?): TensorTFJS

external fun fill(shape: Array<Int>, value: Number, dtype: String): TensorTFJS

external fun scalar(value: Number, dtype: String): TensorTFJS
