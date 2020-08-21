package org.jetbrains.research.kotlin.inference

import org.gradle.api.Project
import org.gradle.api.tasks.SourceSetContainer
import org.jetbrains.kotlin.incremental.isKotlinFile
import java.io.File

internal inline fun <reified T : Any> Project.myExtByName(name: String): T = extensions.getByName(name) as T

internal val Project.mySourceSets: SourceSetContainer
    get() = myExtByName("sourceSets")

internal val Project.myKtSourceSet: Set<File>
    get() = mySourceSets.asMap["main"]!!.allSource.files.filter { it.isKotlinFile(sourceFilesExtensions = listOf("kt")) }.toSet()
