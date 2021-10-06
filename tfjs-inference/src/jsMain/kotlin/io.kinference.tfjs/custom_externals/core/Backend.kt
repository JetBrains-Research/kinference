@file:JsModule("@tensorflow/tfjs-core")
package io.kinference.tfjs.custom_externals.core

import kotlin.js.Promise

open external class KernelBackend

internal external fun registerBackend(name: String, factory: () -> KernelBackend, priority: Int = definedExternally): Boolean

internal external fun removeBackend(name: String)

internal external fun setBackend(backendName: String): Promise<Boolean>

internal external fun getBackend(): String
