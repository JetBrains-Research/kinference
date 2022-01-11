@file:JsModule("@tensorflow/tfjs-backend-cpu")
@file:JsNonModule
package io.kinference.tfjs.externals.backend.cpu

import io.kinference.tfjs.externals.core.KernelBackend

internal external class MathBackendCPU(): KernelBackend
