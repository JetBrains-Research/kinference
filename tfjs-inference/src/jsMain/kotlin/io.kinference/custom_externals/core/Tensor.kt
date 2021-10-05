@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.custom_externals.core

import org.khronos.webgl.*
import kotlin.js.Promise


@JsName("Tensor")
open external class TensorTFJS {
    open var id: Number
    open var dataId: Any
    open var shape: Array<Int>
    open var size: Int
    open var dtype: String /* "float32" | "int32" | "bool" | "complex64" | "string" */
    open var strides: Array<Number>
    open var rank: Int
//    open fun <D : String> buffer(): Promise<TensorBuffer<R, D>>
//    open fun <D : String> bufferSync(): TensorBuffer<R, D>
//    open fun array(): Promise<Any>
//    open fun arraySync(): Any
    open fun data(): Promise<Any>
    open fun dataSync(): dynamic
    open fun bytes(): Promise<dynamic /* Array<Uint8Array> | Uint8Array */>
    open fun print(verbose: Boolean = definedExternally)
    open fun dispose()
}


external fun tensor(values: Float32Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Int32Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Uint8Array, shape: Array<Int>, dtype: String): TensorTFJS

external fun tensor(values: Array<Int>, shape: Array<Int>, dtype: String): TensorTFJS

external fun range(start: Number, stop: Number, step: Number?, dtype: String?): TensorTFJS

external fun fill(shape: Array<Int>, value: Number, dtype: String): TensorTFJS

external fun scalar(value: Number, dtype: String): TensorTFJS
