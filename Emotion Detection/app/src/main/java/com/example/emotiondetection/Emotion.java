package com.example.emotiondetection;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.SparseArray;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import org.tensorflow.lite.Interpreter;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.InputStreamReader;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;


public class Emotion extends Activity {
    boolean flag = false;
    ImageView iv;
    private static final int CAMERA_REQUEST = 1888;
    private static final int PICK_IMAGE = 100;
    FaceDetector detector;
    private ScaleGestureDetector scaleGestureDetector;
    private float mScaleFactor = 1.0f;
    Interpreter interpreter;
    double predict_score = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_emotion);
        String permissions[] = {Manifest.permission.READ_EXTERNAL_STORAGE};
        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleListener());
        if(!hasPermissions(this, permissions)){
            ActivityCompat.requestPermissions(this, permissions, 42);
        }
        try {
            interpreter = new Interpreter(loadTensorModelFile(),null);
            System.out.println("=================================interpreter "+interpreter.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        initClickListner();
    }

    private MappedByteBuffer loadTensorModelFile() throws IOException {
        AssetFileDescriptor asset_FileDescriptor = this.getAssets().openFd("model/model.tflite");
        FileInputStream file_InputStream = new FileInputStream(asset_FileDescriptor.getFileDescriptor());
        FileChannel file_Channel = file_InputStream.getChannel();
        long start_Offset = asset_FileDescriptor.getStartOffset();
        long length = asset_FileDescriptor.getLength();
        MappedByteBuffer buffer = file_Channel.map(FileChannel.MapMode.READ_ONLY,start_Offset,length);
        file_Channel.close();
        return buffer;
    }

    private void initClickListner() {

        iv = (ImageView) findViewById(R.id.imgView);

        Button camera_picture = (Button) findViewById(R.id.camera_picture);
        camera_picture.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
        });

        Button storage_picture = (Button) findViewById(R.id.storage_picture);
        storage_picture.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                openGallery();
            }
        });

        Button logout = (Button) findViewById(R.id.logout);
        logout.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(Emotion.this, MainActivity.class);
                startActivity(intent);
            }
        });

        detector = new FaceDetector.Builder(getApplicationContext())
                .setTrackingEnabled(false)
                .setLandmarkType(FaceDetector.ALL_CLASSIFICATIONS)
                .setClassificationType(FaceDetector.ALL_CLASSIFICATIONS)
                .build();
        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleListener());
    }
    @Override
    public boolean onTouchEvent(MotionEvent motionEvent) {
        scaleGestureDetector.onTouchEvent(motionEvent);
        return true;
    }
    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector scaleGestureDetector) {
            mScaleFactor *= scaleGestureDetector.getScaleFactor();
            mScaleFactor = Math.max(0.1f, Math.min(mScaleFactor, 10.0f));
            iv.setScaleX(mScaleFactor);
            iv.setScaleY(mScaleFactor);
            return true;
        }
    }

    private void openGallery() {
        Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            Uri imageUri = data.getData();
            try {
                Bitmap photo = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                iv.setImageBitmap(photo);
                detectFace(photo);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            iv.setImageBitmap(photo);
            detectFace(photo);
        }
    }

    public String predict(Bitmap capture_photo){
        predict_score = 0;
        Bitmap bitmap = Bitmap.createScaledBitmap(capture_photo, 32, 32, true);
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        float value[][][][] = new float[1][32][32][3];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int pixel = bitmap.getPixel(i, j);
                int red = Color.red(pixel);//(pixel >> 16) & 0xff;
                int green = Color.green(pixel);//(pixel >> 8) & 0xff;
                int blue = Color.blue(pixel);//(pixel) & 0xff;
                float red1 = Float.parseFloat(red + "");
                float blue1 = Float.parseFloat(blue + "");
                float green1 = Float.parseFloat(green + "");
                value[0][j][i][0] = (float) (blue1 / 255.0);
                value[0][j][i][1] = (float) (green1 / 255.0);
                value[0][j][i][2] = (float) (red1 / 255.0);
            }
        }
        float outputs[][] = new float[1][3];
        interpreter.run(value,outputs);
        float out[] = new float[3];
        for(int i=0;i<outputs.length;i++){
            for(int j=0;j<outputs[i].length;j++) {
                out[j] = outputs[i][j];
            }
        }
        int max_index = 0;
        for (int i = 0; i < out.length; i++) {
            max_index = out[i] > out[max_index] ? i : max_index;
            predict_score = out[max_index];
        }
        String output = "Unable to Detect Emotion";
        if(max_index == 0)
            output = "Happy";
        if(max_index == 1)
            output = "Neutral";
        if(max_index == 2)
            output = "Not Satisfied";
        return output;
    }

    public void detectFace(Bitmap photo) {
        Bitmap editedBitmap = Bitmap.createBitmap(photo.getWidth(), photo.getHeight(), photo.getConfig());
        float scale = getResources().getDisplayMetrics().density;
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        paint.setColor(Color.GREEN);
        paint.setShadowLayer(1f, 0f, 1f, Color.WHITE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(0.5f);
        Canvas canvas = new Canvas(editedBitmap);
        canvas.drawBitmap(photo, 0, 0, paint);
        Frame frame = new Frame.Builder().setBitmap(editedBitmap).build();
        SparseArray<Face> faces = detector.detect(frame);
        System.out.println("====================="+faces.size());
        Rect rect = null;

        LinearLayout emotionLayout = findViewById(R.id.emotionLayout);
        emotionLayout.removeAllViews(); // Clear previous views

        for (int index = 0; index < faces.size(); ++index) {
            Face face = faces.valueAt(index);
            rect = new Rect((int)face.getPosition().x, (int)face.getPosition().y, (int)(face.getPosition().x + face.getWidth()), (int)(face.getPosition().y + face.getHeight()));
            Bitmap test_img = cropBitmap(photo, rect);
            String output = predict(test_img);
            canvas.drawRect(face.getPosition().x, face.getPosition().y, face.getPosition().x + face.getWidth(), face.getPosition().y + face.getHeight(), paint);
            canvas.drawText("Face " + (index + 1), face.getPosition().x, face.getPosition().y - 10, paint);

            // Add TextView for each face
            TextView textView = new TextView(this);
            textView.setText("Face " + (index + 1) + ": " + output);
            emotionLayout.addView(textView);
        }

        if (faces.size() == 0) {
            Toast.makeText(Emotion.this, "No Face Detected! Try Again", Toast.LENGTH_LONG).show();
        } else {
            iv.setImageBitmap(editedBitmap);
            emotionLayout.setVisibility(View.VISIBLE); // Show the LinearLayout
        }
    }


    public Bitmap cropBitmap(Bitmap bitmap, Rect rect) {
        int w = rect.right - rect.left;
        int h = rect.bottom - rect.top;
        Bitmap ret = Bitmap.createBitmap(w, h, bitmap.getConfig());
        Canvas canvas = new Canvas(ret);
        canvas.drawBitmap(bitmap, -rect.left, -rect.top, null);
        //bitmap.recycle();
        return ret;
    }


    public boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}