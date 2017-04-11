package io.github.qf6101.tensorflowserving;

import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Created by qfeng on 17-4-7.
 */

/**
 * MNIST Java Client of tensorflow serving example 'https://tensorflow.github.io/serving/serving_basic'
 */
public class MNISTJavaClient {
    private static final Logger logger = LoggerFactory.getLogger(MNISTJavaClient.class.getName());
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    /**
     * Initialize MNIST Java gRPC client
     *
     * @param host host name
     * @param port port
     */
    public MNISTJavaClient(String host, int port) {
        channel = NettyChannelBuilder.forAddress(host, port)
                .usePlaintext(true)
                .maxMessageSize(200 * 1024 * 1024)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    /**
     * Shut down MNIST java gRPC client
     *
     * @throws InterruptedException
     */
    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    /**
     * Predict a batch of MNIST test images
     *
     * @param mnistImageFile MNIST test image file
     * @param mnistLabelFile MNIST test label file
     */
    public void predict(String mnistImageFile, String mnistLabelFile) {
        List<int[][]> images = MnistReader.getImages(mnistImageFile);
        int[] labels = MnistReader.getLabels(mnistLabelFile);

        for (int i = 0; i < 10; ++i) {
            TensorProto imageTensor = createImageTensor(images.get(i));
            if(imageTensor != null) requestService(imageTensor, labels[i]);
        }
    }

    /**
     * Create image tensor from image content
     *
     * @param image image content
     * @return image tensor
     */
    private TensorProto createImageTensor(int[][] image) {
        try {
            TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder()
                    .setSize(1).build();
            TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder()
                    .setSize(image.length * image.length).build();
            TensorShapeProto imageFeatureShape = TensorShapeProto.newBuilder()
                    .addDim(featuresDim1).addDim(featuresDim2).build();

            TensorProto.Builder imageTensorBuilder = TensorProto.newBuilder();
            imageTensorBuilder.setDtype(DataType.DT_FLOAT).setTensorShape(imageFeatureShape);

            for (int i = 0; i < image.length; ++i) {
                for (int j = 0; j < image.length; ++j) {
                    imageTensorBuilder.addFloatVal(image[i][j]);
                }
            }

            return imageTensorBuilder.build();
        } catch (Exception ex) {
            logger.error("Create image tensor fail!", ex);
            return null;
        }
    }

    /**
     * Request MNIST classification service
     *
     * @param imagesTensorProto request image tensor
     * @param label request image label
     */
    private void requestService(TensorProto imagesTensorProto, int label) {
        // generate MNIST classification gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder()
                .setValue(1)
                .build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder()
                .setName("mnist")
                .setVersion(version)
                .setSignatureName("predict_images")
                .build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder()
                .setModelSpec(modelSpec)
                .putInputs("images", imagesTensorProto)
                .build();

        // request MNIST classification gRPC server
        Predict.PredictResponse response;
        try {
            response = blockingStub.withDeadlineAfter(10, TimeUnit.SECONDS).predict(request);
            TensorProto scores = response.getOutputsMap().get("scores");
            System.out.println("label: " + label + ", predicted : " + scores.getFloatValList());
        } catch (Exception ex) {
            logger.error("request RPC failed: {0}", ex);
            return;
        }
    }
}
