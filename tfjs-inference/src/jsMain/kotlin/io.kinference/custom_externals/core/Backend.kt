@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.custom_externals.core

import kotlin.js.Promise

open external class KernelBackend

external fun registerBackend(name: String, factory: () -> KernelBackend, priority: Int = definedExternally): Boolean

external fun removeBackend(name: String)

external fun setBackend(backendName: String): Promise<Boolean>

external fun getBackend(): String

external fun enableDebugMode()
