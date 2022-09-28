@file:JsModule("@tensorflow/tfjs-backend-webgl")
@file:JsNonModule
package io.kinference.backend.webgl

import io.kinference.backend.KernelBackend

internal external class MathBackendWebGL(gpgpu: dynamic) : KernelBackend
