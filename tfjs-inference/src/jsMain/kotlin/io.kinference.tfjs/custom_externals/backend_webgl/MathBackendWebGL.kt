@file:JsModule("@tensorflow/tfjs-backend-webgl")
package io.kinference.tfjs.custom_externals.backend_webgl

import io.kinference.tfjs.custom_externals.core.KernelBackend

internal external class MathBackendWebGL(gpgpu: dynamic) : KernelBackend
