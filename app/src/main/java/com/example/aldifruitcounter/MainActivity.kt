package com.example.aldifruitcounter

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.Manifest
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var captureButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter
    private val cameraRequestCode = 100
    private val storageRequestCode = 200

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        captureButton = findViewById(R.id.captureButton)
        resultTextView = findViewById(R.id.resultTextView)

        // Load TensorFlow Lite model
        tflite = Interpreter(loadModelFile())

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), storageRequestCode)
        }

        // Initialize Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraRequestCode)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build()
                .also { it.setSurfaceProvider(viewFinder.surfaceProvider) }

            imageCapture = ImageCapture.Builder().build()

            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
        }, ContextCompat.getMainExecutor(this))

        captureButton.setOnClickListener { captureAndAnalyzeImage() }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("new_set.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun captureAndAnalyzeImage() {
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(createTempFile()).build()
        imageCapture.takePicture(outputFileOptions, ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bitmap = outputFileResults.savedUri?.let { loadBitmapFromFile(it) } // Convert to Bitmap
                    if (bitmap != null) {
                        analyzeImage(bitmap)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("FruitCounter", "Image capture failed: ${exception.message}", exception)
                }
            })
    }

    private fun loadBitmapFromFile(uri: Uri): Bitmap {
        val source = ImageDecoder.createSource(contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
            // Force the bitmap to be decoded with software rendering and ARGB_8888 format
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE)
        }
    }

    private fun analyzeImage(bitmap: Bitmap) {
        // Resize bitmap to match model input size (640x640)
        val inputImage = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // Convert to ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4) // 640x640x3, 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())  // Set correct byte order
        inputBuffer.rewind()

        for (y in 0 until 640) {
            for (x in 0 until 640) {
                val pixel = inputImage.getPixel(x, y)
                inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f) // Red
                inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)  // Green
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)       // Blue
            }
        }

        // Run inference
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 67, 8400), DataType.FLOAT32) // Output shape [1, 67, 8400]
        tflite.run(inputBuffer, outputBuffer.buffer.rewind())  // Running inference

        // Get the result from the output buffer
        val result = outputBuffer.floatArray

        // Debugging: Print all class outputs
        Log.d("FruitCounter", "Model Output (First 10 values):")
        for (i in 0 until 10) {  // Print the first 10 class values for debugging
            Log.d("FruitCounter", "Class $i: ${result[i]}")
        }

        // Store the count of recognized items
        val recognizedItems = mutableMapOf<String, Int>()

        // Identify the most recognized item and its count
        var itemName = ""
        var itemCount = 0

        for (i in 0 until 67) {
            // Extract the 8400 values for each class (item)
            val classValues = result.slice(i * 8400 until (i + 1) * 8400)
            val maxValue = classValues.maxOrNull() ?: 0f  // Find the highest value
            val maxIndex = classValues.indexOf(maxValue)  // Get the index of the max value

            // Get the class name using the map instead of list
            val className = getClassName(i)  // Get the class name based on the class index

            // If the value is high enough, we consider it recognized (use a threshold for confidence)
            if (maxValue > 0.5) {  // Threshold for recognition
                // Check if this item was already recognized and increment count
                recognizedItems[className] = recognizedItems.getOrDefault(className, 0) + 1
            }
        }

        // Find the most recognized fruit and its count
        if (recognizedItems.isNotEmpty()) {
            val mostRecognizedItem = recognizedItems.maxByOrNull { it.value }
            itemName = mostRecognizedItem?.key ?: "Unknown"
            itemCount = mostRecognizedItem?.value ?: 0
        }

        // Build result string in the desired format
        val itemRecognitionResult = StringBuilder()
        itemRecognitionResult.append("Item Recognized: $itemName\n")
        itemRecognitionResult.append("Count of Items: $itemCount")

        // Show the result in a TextView
        resultTextView.text = itemRecognitionResult.toString()
    }


    // Helper function to get class names, now using a map
    private fun getClassName(classIndex: Int): String {
        val classNames = mapOf(
            0 to "Apple Braeburn", 1 to "Apple Crimson Snow", 2 to "Apple Golden 1", 3 to "Apple Golden 2", 4 to "Apple Golden 3",
            5 to "Apple Granny Smith", 6 to "Apple Pink Lady", 7 to "Apple Red 1", 8 to "Apple Red 2", 9 to "Apple Red 3",
            10 to "Apple Red Delicious", 11 to "Apple Red Yellow 1", 12 to "Apple Red Yellow 2", 13 to "Apricot",
            14 to "Avocado", 15 to "Avocado ripe", 16 to "Banana", 17 to "Banana Lady Finger", 18 to "Banana Red",
            19 to "Beetroot", 20 to "Blueberry", 21 to "Cactus fruit", 22 to "Cantaloupe 1", 23 to "Cantaloupe 2",
            24 to "Carambula", 25 to "Cauliflower", 26 to "Cherry 1", 27 to "Cherry 2", 28 to "Cherry Rainier",
            29 to "Cherry Wax Black", 30 to "Cherry Wax Red", 31 to "Cherry Wax Yellow", 32 to "Chestnut",
            33 to "Clementine", 34 to "Cocos", 35 to "Corn", 36 to "Corn Husk", 37 to "Cucumber Ripe",
            38 to "Cucumber Ripe 2", 39 to "Dates", 40 to "Eggplant", 41 to "Fig", 42 to "Ginger Root", 43 to "Granadilla",
            44 to "Grape Blue", 45 to "Grape Pink", 46 to "Grape White", 47 to "Grape White 2", 48 to "Grape White 3",
            49 to "Grape White 4", 50 to "Grapefruit Pink", 51 to "Grapefruit White", 52 to "Guava", 53 to "Hazelnut",
            54 to "Huckleberry", 55 to "Kaki", 56 to "Kiwi", 57 to "Kohlrabi", 58 to "Kumquats", 59 to "Lemon",
            60 to "Lemon Meyer", 61 to "Limes", 62 to "Lychee", 63 to "Mandarine", 64 to "Mango", 65 to "Mango Red",
            66 to "Mangostan", 67 to "Maracuja", 68 to "Melon Piel de Sapo", 69 to "Mulberry", 70 to "Nectarine",
            71 to "Nectarine Flat", 72 to "Nut Forest", 73 to "Nut Pecan", 74 to "Onion Red", 75 to "Onion Red Peeled",
            76 to "Onion White", 77 to "Orange", 78 to "Papaya", 79 to "Passion Fruit", 80 to "Peach", 81 to "Peach 2",
            82 to "Peach Flat", 83 to "Pear", 84 to "Pear 2", 85 to "Pear Abate", 86 to "Pear Forelle", 87 to "Pear Kaiser",
            88 to "Pear Monster", 89 to "Pear Red", 90 to "Pear Stone", 91 to "Pear Williams", 92 to "Pepino",
            93 to "Pepper Green", 94 to "Pepper Orange", 95 to "Pepper Red", 96 to "Pepper Yellow", 97 to "Physalis",
            98 to "Physalis with Husk", 99 to "Pineapple", 100 to "Pineapple Mini", 101 to "Pitahaya Red",
            102 to "Plum", 103 to "Plum 2", 104 to "Plum 3", 105 to "Pomegranate", 106 to "Pomelo Sweetie",
            107 to "Potato Red", 108 to "Potato Red Washed", 109 to "Potato Sweet", 110 to "Potato White", 111 to "Quince",
            112 to "Rambutan", 113 to "Raspberry", 114 to "Redcurrant", 115 to "Salak", 116 to "Strawberry", 117 to "Strawberry 2",
            118 to "Strawberry 3", 119 to "Strawberry 4", 120 to "Tamarillo", 121 to "Tangerine", 122 to "Tomato",
            123 to "Tomato Cherry", 124 to "Tomato Marzano", 125 to "Tomato Red", 126 to "Tomato Yellow", 127 to "Watermelon"
        )
        return classNames[classIndex] ?: "Unknown"
    }
}
