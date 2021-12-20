package io.kinference.utils.wgpu.generation

import io.kinference.utils.wgpu.generation.grammar.CBaseVisitor
import io.kinference.utils.wgpu.generation.grammar.CParser
import io.kinference.utils.wgpu.generation.models.*
import io.kinference.utils.wgpu.generation.utils.terminalNodes

class WgpuDeclarationCollector : CBaseVisitor<Unit>() {
    private val enumDeclarations: MutableList<EnumDeclaration> = ArrayList()
    private val structDeclarations: MutableList<StructDeclaration> = ArrayList()
    private val typeAliases: MutableMap<String, TypeRepresentation> = HashMap()

    fun getCollectedDeclarations(): LibraryDeclarations =
        LibraryDeclarations(enumDeclarations, structDeclarations, typeAliases)

    override fun visitDeclaration(ctx: CParser.DeclarationContext) {
        val specifiers = ctx.declarationSpecifiers().declarationSpecifier()
        if (specifiers.getOrNull(0)?.storageClassSpecifier()?.Typedef() != null) {
            specifiers.getOrNull(1)?.typeSpecifier()?.let { typeSpecifier ->
                typeSpecifier.structOrUnionSpecifier()?.let { structSpecifier ->
                    processStructSpecifier(structSpecifier)?.let { structDeclaration ->
                        structDeclarations.add(structDeclaration)
                    }
                } ?: typeSpecifier.enumSpecifier()?.let { enumSpecifier ->
                    enumDeclarations.add(processEnumSpecifier(enumSpecifier))
                } ?: specifiers.getOrNull(2)?.text?.let { alias ->
                    val type = processTypeSpecifier(typeSpecifier)
                    typeAliases[alias] = type
                }
            }
        }
        super.visitDeclaration(ctx)
    }

    private fun processEnumSpecifier(ctx: CParser.EnumSpecifierContext): EnumDeclaration =
        EnumDeclaration(
            name = ctx.Identifier().text,
            options = ctx.enumeratorList().enumerator().map {
                EnumOption(
                    name = it.enumerationConstant().text,
                    value = it.constantExpression().text
                )
            }
        )

    private fun processStructSpecifier(ctx: CParser.StructOrUnionSpecifierContext): StructDeclaration? {
        require(ctx.structOrUnion().Struct() != null) { "Unions are not supported" }
        val declarations = ctx.structDeclarationList()?.structDeclaration() ?: return null
        return StructDeclaration(
            name = ctx.Identifier().text,
            fields = declarations.map { processStructDeclaration(it) }
        )
    }

    private fun processStructDeclaration(ctx: CParser.StructDeclarationContext): StructField {
        val nodes = ctx.terminalNodes()
            .map { it.text }
            .filterNot { it in listOf("struct", "const", ";") }
        require(!nodes.contains(",")) { "Multiple declarations in one line are not supported" }
        val typeName = nodes.first();
        val pointer = nodes.contains("*");
        val name = nodes.last()
        return StructField(name, TypeRepresentation(typeName, pointer))
    }

    private fun processTypeSpecifier(ctx: CParser.TypeSpecifierContext): TypeRepresentation =
        ctx
            .structOrUnionSpecifier()?.let {
                TypeRepresentation(it.Identifier().text, false)
            } ?: ctx.typeSpecifier()?.let {
                TypeRepresentation(it.text, true)
            } ?: TypeRepresentation(ctx.text, false)
}
