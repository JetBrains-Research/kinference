package io.kinference.gradle

import org.gradle.api.file.SourceDirectorySet
import org.gradle.api.internal.HasConvention
import org.gradle.api.tasks.SourceSet
import org.jetbrains.kotlin.gradle.plugin.KotlinSourceSet

const val generatedDir = "src/main/kotlin-gen"

val SourceSet.kotlin: SourceDirectorySet
    get() = (this as HasConvention)
        .convention
        .getPlugin(KotlinSourceSet::class.java)
        .kotlin
