package io.kinference.model

import java.io.File

actual fun Model.Companion.load(file: String): Model = load(File(file).readBytes())
