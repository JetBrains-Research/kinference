@file:JsModule("@tensorflow/tfjs-backend-webgl")
package io.kinference.tfjs.externals.backend.webgl

import io.kinference.tfjs.externals.core.KernelBackend

internal external class MathBackendWebGL(gpgpu: dynamic) : KernelBackend
