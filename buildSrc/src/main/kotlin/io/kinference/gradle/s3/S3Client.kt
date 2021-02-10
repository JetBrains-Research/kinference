package io.kinference.gradle.s3

import com.amazonaws.auth.AWSStaticCredentialsProvider
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.*
import java.io.*


object S3Client {
    private const val bucket: String = "kinference.grazie.aws.intellij.net"

    private val client by lazy {
        AmazonS3ClientBuilder.standard()
            .withRegion("eu-west-1")
            .withCredentials(AWSStaticCredentialsProvider(BasicAWSCredentials(Config.awsAccessKey, Config.awsSecretKey)))
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
        val objects = getObjects(prefix).filter { !it.key.endsWith("/") }
        val toDownload = objects.filter {
            val toKey = it.key.drop(prefix.length)
            val toFile = File(toFolder, toKey)

            //Have to use variant with size because of multipart etag in s3
            !toFile.exists() || toFile.length() != it.size
        }

        if (toDownload.isEmpty()) return

        println("Downloading missing files from s3://$bucket/$prefix")
        for (obj in toDownload) {
            val toKey = obj.key.drop(prefix.length)
            val toFile = File(toFolder, toKey)

            if (toFile.exists()) toFile.delete()

            println("Downloading file from key ${obj.key}")

            toFile.parentFile.mkdirs()

            toFile.outputStream().use { output ->
                client.getObject(bucket, obj.key).objectContent.use { input -> input.transferTo(output) }
            }
        }

        println("Finished files downloading")
    }
}

