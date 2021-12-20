package io.kinference.utils.wgpu.generation

import com.squareup.kotlinpoet.AnnotationSpec
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.TypeAliasSpec
import io.kinference.utils.wgpu.generation.generators.generateEnumDeclaration
import io.kinference.utils.wgpu.generation.generators.generateStructDeclaration
import io.kinference.utils.wgpu.generation.grammar.CLexer
import io.kinference.utils.wgpu.generation.grammar.CParser
import io.kinference.utils.wgpu.generation.models.*
import io.kinference.utils.wgpu.generation.utils.TypeMapper
import jnr.ffi.Pointer
import org.antlr.v4.runtime.BailErrorStrategy
import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import java.io.File

class WgpuJnrGenerator(
    private val headers: List<String>,
    private val sourceDirectory: File,
    private val packageName: String,
) {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            if (args.size < 3) {
                error("arguments: headers-directory source-directory package-name")
            }
            val headersDirectory = File(args[0])
            val sourceDirectory = File(args[1])
            val packageName = args[2]

            sourceDirectory.deleteRecursively()

            WgpuJnrGenerator(
                headersDirectory.listFiles()!!.map { it.readText() },
                sourceDirectory,
                packageName
            ).generate()
        }
    }

    fun generate() {
        val outputDirectory = sourceDirectory.resolve(packageName.replace(".", File.separator))
        outputDirectory.mkdirs()

        val collector = WgpuDeclarationCollector()
        headers.forEach { header ->
            val lexer = CLexer(CharStreams.fromString(header))
            val parser = CParser(CommonTokenStream(lexer))
            parser.errorHandler = BailErrorStrategy()
            parser.compilationUnit().accept(collector)
        }
        val declarations = collector.getCollectedDeclarations()

        generateEnums(declarations.enumDeclarations, outputDirectory)
        generateStructs(declarations, outputDirectory)
        generateTypeAliases(declarations, outputDirectory)
    }

    private fun generateEnums(enumDeclarations: List<EnumDeclaration>, outputDirectory: File) {
        val enumsFileName = "WGPUEnums"
        val enumsFile = baseFileBuilder(enumsFileName)
            .apply {
                enumDeclarations.forEach { enumDeclaration ->
                    addType(generateEnumDeclaration(packageName, enumDeclaration))
                }
            }
            .build()
            .toString()
        outputDirectory
            .resolve("$enumsFileName.kt")
            .writeText(enumsFile)
    }

    private fun generateStructs(libraryDeclarations: LibraryDeclarations, outputDirectory: File) {
        val typeMapper = TypeMapper(libraryDeclarations)

        libraryDeclarations.structDeclarations.forEach { structDeclaration ->
            val structFile = baseFileBuilder(structDeclaration.name)
                .apply {
                    addAliasedImport(Boolean::class, "KBoolean")
                    addAliasedImport(Float::class, "KFloat")
                    addAliasedImport(Double::class, "KDouble")
                    addAliasedImport(Pointer::class, "JNRPointer")

                    addType(generateStructDeclaration(packageName, structDeclaration, typeMapper))
                }
                .build()
                .toString()
            outputDirectory
                .resolve("${structDeclaration.name}.kt")
                .writeText(structFile)
        }
    }

    private fun generateTypeAliases(libraryDeclarations: LibraryDeclarations, outputDirectory: File) {
        val typeMapper = TypeMapper(libraryDeclarations)

        val aliasesFileName = "WGPUTypeAliases"
        val aliasesFile = baseFileBuilder(aliasesFileName)
            .apply {
                addTypeAlias(TypeAliasSpec.builder("CString", Pointer::class).build())
                libraryDeclarations.structDeclarations.forEach { declaration ->
                    addTypeAlias(TypeAliasSpec.builder("${declaration.name}Pointer", Pointer::class).build())
                }

                libraryDeclarations.typeAliases.keys.sorted().forEach { alias ->
                    when (val fieldType = typeMapper.getType(TypeRepresentation(alias, false))) {
                        BooleanType -> addTypeAlias(TypeAliasSpec.builder(alias, Boolean::class).build())
                        Unsigned16Type -> addTypeAlias(TypeAliasSpec.builder(alias, Int::class).build())
                        Unsigned32Type -> addTypeAlias(TypeAliasSpec.builder(alias, Long::class).build())
                        Unsigned64Type -> addTypeAlias(TypeAliasSpec.builder(alias, Long::class).build())
                        Signed16Type -> addTypeAlias(TypeAliasSpec.builder(alias, Short::class).build())
                        Signed32Type -> addTypeAlias(TypeAliasSpec.builder(alias, Int::class).build())
                        Signed64Type -> addTypeAlias(TypeAliasSpec.builder(alias, Long::class).build())
                        FloatType -> addTypeAlias(TypeAliasSpec.builder(alias, Float::class).build())
                        DoubleType -> addTypeAlias(TypeAliasSpec.builder(alias, Double::class).build())
                        PointerType, CStringType, is StructPointerType ->
                            addTypeAlias(TypeAliasSpec.builder(alias, Pointer::class).build())
                        is StructType -> addTypeAlias(TypeAliasSpec.builder(alias, ClassName(packageName, fieldType.name)).build())
                        is EnumType -> addTypeAlias(TypeAliasSpec.builder(alias, ClassName(packageName, fieldType.name)).build())
                    }
                }
            }
            .build()
            .toString()
        outputDirectory.resolve("$aliasesFileName.kt").writeText(aliasesFile)
    }

    private fun baseFileBuilder(fileName: String): FileSpec.Builder =
        FileSpec.builder(packageName, fileName)
            .indent("    ")
            .addAnnotation(
                AnnotationSpec.builder(Suppress::class)
                    .addMember("%S", "PropertyName")
                    .addMember("%S", "RedundantVisibilityModifier")
                    .addMember("%S", "Unused")
                    .addMember("%S", "UnusedImport")
                    .build()
            )
}
