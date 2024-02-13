package io.kinference.gradle.s3

import com.amazonaws.auth.AWSStaticCredentialsProvider
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.S3ObjectSummary
import java.io.File


class S3Client(val config: Config) {
    private val bucket: String = "kinference.dev.grazie.aws.intellij.net"

    private val client by lazy {
        AmazonS3ClientBuilder.standard()
            .withRegion("eu-west-1")
            .withCredentials(AWSStaticCredentialsProvider(BasicAWSCredentials(config.awsAccessKey, config.awsSecretKey)))
            .build()
    }

    private fun getObjects(prefix: String): List<S3ObjectSummary> {
        val objects = ArrayList<S3ObjectSummary>()
        var isFirst = true
        var objectListing = client.listObjects(bucket)
        while (objectListing.isTruncated || isFirst) {
            if (!isFirst) {
                objectListing = client.listNextBatchOfObjects(objectListing)
            }
            objects.addAll(objectListing.objectSummaries.filter {
                it.key.startsWith(prefix)
            })
            isFirst = false
        }
        return objects
    }

    fun copyObjects(prefix: String, toFolder: File) {
        toFolder.mkdirs()

        val objects = getObjects(prefix).filter { !it.key.endsWith("/") }
        val files = objects.map {
            val toKey = it.key.drop(prefix.length)
            File(toFolder, toKey)
        }

//        File(toFolder, "descriptor.txt").writeText(
//            files.filter { it.name != "model.onnx" }.joinToString(separator = "\n") {
//                it.absolutePath.drop(toFolder.absolutePath.length + 1).replace('\\', '/')
//            }
//        )

        val toDownload = objects.zip(files).filter { (obj, file) ->
            //Have to use variant with size because of multipart etag in s3
            !file.exists() || file.length() != obj.size
        }

        if (toDownload.isEmpty()) return

        println("Downloading missing files from s3://$bucket/$prefix")
        for ((obj, file) in toDownload) {
            if (file.exists()) file.delete()

            println("Downloading file from key ${obj.key}")

            file.parentFile.mkdirs()

            file.outputStream().use { output ->
                client.getObject(bucket, obj.key).objectContent.use { input -> input.transferTo(output) }
            }
        }

        println("Finished files downloading")
    }
}

