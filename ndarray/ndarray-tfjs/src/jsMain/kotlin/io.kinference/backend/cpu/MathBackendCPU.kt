@file:JsModule("@tensorflow/tfjs-backend-cpu")
@file:JsNonModule
package io.kinference.backend.cpu

import io.kinference.backend.KernelBackend

internal external class MathBackendCPU(): KernelBackend
