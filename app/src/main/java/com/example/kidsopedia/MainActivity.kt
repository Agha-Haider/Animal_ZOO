package com.example.kidsopedia

import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.MediaController
import android.widget.TextView
import com.example.kidsopedia.ml.Models
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    lateinit var selectButton:Button
    lateinit var predictButton: Button
    lateinit var imageView: ImageView
    lateinit var result: TextView
    lateinit var bitmap: Bitmap


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        selectButton=findViewById(R.id.select)
        predictButton=findViewById(R.id.predict)
        imageView=findViewById(R.id.image_View)
        result=findViewById(R.id.resView)
        var labels=application.assets.open("descriptiontext.txt").bufferedReader().readLines()
        var imageProcessor=ImageProcessor.Builder()
            .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR))
            .build()

        selectButton.setOnClickListener{
            var intent:Intent= Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent,100)
        }
        predictButton.setOnClickListener{
            var tensorImage=TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage= imageProcessor.process(tensorImage)
            val model = Models.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)

            inputFeature0.loadBuffer(tensorImage.buffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
            var maxIdx=0
            outputFeature0.forEachIndexed { index, fl ->
                if(outputFeature0[maxIdx]<fl){
                    maxIdx=index

                }
            }
            result.setText(labels[maxIdx])

// Releases model resources if no longer used.
            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==100){
            var uri=data?.data
            bitmap=MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            imageView.setImageBitmap(bitmap)
        }
    }
}